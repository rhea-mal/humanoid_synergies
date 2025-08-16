/**
 * @file simulation.cpp
 * @author Rhea Malhotra(rheamal@stanford.edu)
 * @brief
 * @version 0.2
 * @date 2024-05-28
 *
 * @copyright Copyright (c) 2024
 *
 */

// c++ includes 
#include <math.h>
#include <signal.h>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <random>

// sai includes 
#include "SaiGraphics.h"
#include "SaiModel.h"
#include "SaiSimulation.h"
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"
#include "RobotController.h"

// project includes 
#include "../include/Human.h"
#include "redis_keys.h"
#include <yaml-cpp/yaml.h>
#include "../include/CLI11.hpp"

bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;
using namespace Optitrack;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// specify urdf and robots
const string robot_file = "./resources/model/HRP4c.urdf";
string robot_name = "HRP4C";  // will be added with suffix of robot id 
const string camera_name = "camera_fixed";
const std::string yaml_fname = "./resources/controller_settings_multi_dancers.yaml";

// globals
VectorXd nominal_posture;
VectorXd control_torques;
VectorXd ui_torques;
mutex mutex_update;
mutex mutex_torques;

// data structures 
struct OptitrackData {
    std::map<std::string, int> body_index_mapping;  // map body part name to index from redis database
    std::map<std::string, Affine3d> current_pose;  // current optitrack pose
    std::vector<int> human_ids;  // human id prefixes 
};

struct SimBodyData {
    std::map<std::string, Affine3d> starting_pose;
    std::map<std::string, Affine3d> current_pose;
};

struct ControllerData {
    std::vector<std::string> control_links;
    std::vector<Vector3d> control_points;
};

double MOTION_SF = 0.9;
const double MAX_RADIUS_ARM = 0.5;  // saturate arms within a sphere distance of the pelvis 

enum State {
    INIT = 0,
    CALIBRATION,
    TRACKING,
    TEST,
    RESET
};

// this is OT R transpose only
Eigen::Matrix3d quaternionToRotationMatrix(const VectorXd& quat) {
    Eigen::Quaterniond q;
    q.x() = quat[0];
    q.y() = quat[1];
    q.z() = quat[2];
    q.w() = quat[3];
    return q.toRotationMatrix();
}

Eigen::Matrix3d reOrthogonalizeRotationMatrix(const Eigen::Matrix3d& mat) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d correctedMat = svd.matrixU() * svd.matrixV().transpose();
    return correctedMat;
}

void control(std::shared_ptr<Human> human,
             std::shared_ptr<SaiModel::SaiModel> robot,
             OptitrackData& optitrack_data,
             SimBodyData& sim_body_data,
             ControllerData& controller_data,
             const VectorXd& q_init);

Eigen::VectorXd generateRandomVector(double lowerBound, double upperBound, int size) {
    // Initialize a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(lowerBound, upperBound);

    // Generate random vector
    Eigen::VectorXd randomVec(size);
    for (int i = 0; i < size; ++i) {
        randomVec(i) = dis(gen);
    }

    return randomVec;
}

template <typename T>
int sign(T value) {
    if (value > 0) {
        return 1;  // Positive
    } else if (value < 0) {
        return -1; // Negative
    } else {
        return 0;  // Zero
    }
}

Matrix3d R_camera_world = Matrix3d::Identity();
Matrix3d R_mirror;
double Z_ROTATION = 0 * M_PI / 180;
Matrix3d R_optitrack_to_sai;
bool DEBUG = false;
std::string NAME = "Hannah";
int ROBOT_ID = 0;

double MAX_TORQUE_SPIKE = 50;
double MAX_JOINT_TORQUE = 1000;

int main(int argc, char** argv) {

    CLI::App app("Argument parser app");
    argv = app.ensure_utf8(argv);

    app.add_option("--name", NAME, "Name (Tracy or Hannah or User)");
    app.add_option("--sf", MOTION_SF, "Motion Scale Factor");

    CLI11_PARSE(app, argc, argv);

    std::cout << "User Name: " << NAME << "\n";
    std::cout << "Motion Scaling: " << MOTION_SF << "\n";

    // parse input 
    if (NAME == "Hannah") {
        ROBOT_ID = 0;
        // MOTION_SF = 0.9;
    } else if (NAME == "Tracy") {
        ROBOT_ID = 1;
        // MOTION_SF = 0.9;
    } else if (NAME == "User") {
        ROBOT_ID = 2;
    }

    if (NAME != "Hannah" && NAME != "Tracy" && NAME != "User") {
        throw runtime_error("Must select either Hannah or Tracy or User");
    }

    robot_name += std::to_string(ROBOT_ID);
    
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // initialize redis 
    redis_client.setInt(MULTI_USER_READY_KEY[ROBOT_ID], 0);

    // setup struct 
    auto optitrack_data = OptitrackData();
    auto sim_body_data = SimBodyData();
    auto controller_data = ControllerData();

    // parse yaml controller settings 
    YAML::Node config = YAML::LoadFile(yaml_fname);

    // optitrack settings 
    YAML::Node current_node = config["optitrack"];
    std::vector<std::string> body_part_names = current_node["body_part_names"].as<std::vector<std::string>>();
    std::vector<int> optitrack_ids = current_node["indices"].as<std::vector<int>>();
    std::vector<int> human_ids = current_node["human_ids"].as<std::vector<int>>();
    DEBUG = current_node["debug"].as<bool>();

    Z_ROTATION = current_node["z_rotation"].as<double>();
    std::cout << "Z rotation (degrees): " << Z_ROTATION << "\n";
    // R_optitrack_to_sai = (AngleAxisd(Z_ROTATION * M_PI / 180, Vector3d::UnitZ()) * AngleAxisd(0 * M_PI / 2, Vector3d::UnitX())).toRotationMatrix();
    R_optitrack_to_sai = Matrix3d::Identity();

    // // // // MOTION GPT MAPPING
    // double deg2rad = M_PI / 180.0;
    // double x_rot = 90.0 * deg2rad;
    // double z_rot = 90.0 * deg2rad;

    // // Rotation: first X, then Z (right-multiplied)
    // R_optitrack_to_sai =
    //     AngleAxisd(z_rot, Vector3d::UnitZ()) *
    //     AngleAxisd(x_rot, Vector3d::UnitX());

    // create map between body part names and indices 
    std::map<std::string, int> body_index_mapping; 
    for (int i = 0; i < body_part_names.size(); ++i) {
        body_index_mapping[body_part_names[i]] = optitrack_ids[i];
    }

    optitrack_data.body_index_mapping = body_index_mapping;    

    // create map between body part names and transformations
    for (int i = 0; i < body_part_names.size(); ++i) {
        optitrack_data.current_pose[body_part_names[i]] = Affine3d::Identity();
    }

    optitrack_data.human_ids = human_ids;

    // controller settings 
    current_node = config["controller"];
    controller_data.control_links = current_node["links"].as<std::vector<std::string>>();
    std::vector<double> control_offsets = current_node["points"].as<std::vector<double>>();
    for (auto val : control_offsets) {
        controller_data.control_points.push_back(Vector3d(0, 0, val));
    }

    // DEBUG
    // #ifdef DEBUG
    std::cout << "Optitrack Tracking Information\n---\n";
    for (auto it = optitrack_data.body_index_mapping.begin(); it != optitrack_data.body_index_mapping.end(); ++it) {
        std::cout << "Body: " << it->first << " with index " << it->second << "\n";
    }

    std::cout << "\nController Settings\n---\n";
    for (int i = 0; i < controller_data.control_links.size(); ++i) {
        std::cout << "Control link " << controller_data.control_links[i] << " with control point " << controller_data.control_points[i].transpose() << "\n";
    }
    // #endif

    // set initial z rotation viewing angle to be 0 (adjusted later if needed)
    redis_client.setDouble(Z_VIEWING_ANGLE, 180);
    // R_optitrack_to_sai *= AngleAxisd(90 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix();

    auto robot = make_shared<SaiModel::SaiModel>(robot_file, false);
    robot->updateModel();

    // create human class for tracking 
    auto human = std::make_shared<Optitrack::Human>(controller_data.control_links);
    human->setMultiRotationReference(controller_data.control_links, {controller_data.control_links.size(), R_optitrack_to_sai});

    // start controller thread
    thread control_thread(control, human, robot, std::ref(optitrack_data), std::ref(sim_body_data), std::ref(controller_data), nominal_posture);

    control_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void control(std::shared_ptr<Optitrack::Human> human,
             std::shared_ptr<SaiModel::SaiModel> robot,
             OptitrackData& optitrack_data,
             SimBodyData& sim_body_data,
             ControllerData& controller_data,
             const VectorXd& q_init) {

    // create redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // reset user ready key 
    redis_client.setInt(MULTI_USER_READY_KEY[ROBOT_ID], 0);
    redis_client.setInt(MULTI_RESET_ROBOT_KEY[ROBOT_ID], 0);

	// update robot model and initialize control vectors
    VectorXd robot_q = redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[ROBOT_ID]);
    VectorXd robot_dq = redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[ROBOT_ID]);
    robot->setQ(robot_q);
    robot->setDq(robot_dq);
    robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

    // create task mappings 
    std::map<std::string, std::shared_ptr<SaiPrimitives::MotionForceTask>> tasks;

    std::vector<Vector3d> fully_controlled_directions = {Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ()};
    std::vector<Vector3d> yz_controlled_directions = {Vector3d::UnitY(), Vector3d::UnitZ()};
    std::vector<Vector3d> uncontrolled_directions = {};

    for (int i = 0; i < controller_data.control_links.size(); ++i) {
        Affine3d compliant_frame = Affine3d::Identity();
        compliant_frame.translation() = controller_data.control_points[i];

        if (controller_data.control_links[i] == "trunk_rz") {
            // 2 DOF task
            tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                        controller_data.control_links[i],
                                                                                                        uncontrolled_directions,
                                                                                                        fully_controlled_directions,
                                                                                                        compliant_frame,
                                                                                                        controller_data.control_links[i]);

            tasks[controller_data.control_links[i]]->disableInternalOtg();
            tasks[controller_data.control_links[i]]->disableSingularityHandling();
            tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
            tasks[controller_data.control_links[i]]->setOriControlGains(350, 25, 0);

        } else if (controller_data.control_links[i] == "neck_link2") {
            // 2 DOF task 
            tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                        controller_data.control_links[i],
                                                                                                        uncontrolled_directions,
                                                                                                        fully_controlled_directions,
                                                                                                        compliant_frame,
                                                                                                        controller_data.control_links[i]);

            tasks[controller_data.control_links[i]]->disableInternalOtg();
            tasks[controller_data.control_links[i]]->disableSingularityHandling();
            tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
            tasks[controller_data.control_links[i]]->setOriControlGains(350, 25, 0);

        } else if (controller_data.control_links[i] == "ra_link4" || controller_data.control_links[i] == "la_link4") {
            // 1 DOF task
            tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                        controller_data.control_links[i],
                                                                                                        uncontrolled_directions,
                                                                                                        fully_controlled_directions,
                                                                                                        compliant_frame,
                                                                                                        controller_data.control_links[i]);
            tasks[controller_data.control_links[i]]->disableInternalOtg();
            tasks[controller_data.control_links[i]]->disableSingularityHandling();
            tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
            tasks[controller_data.control_links[i]]->setOriControlGains(350, 25, 0);

        } else if (controller_data.control_links[i] == "LL_KOSY_L56" || controller_data.control_links[i] == "RL_KOSY_L56") {
            // 1 DOF task
            tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                        controller_data.control_links[i],
                                                                                                        uncontrolled_directions,
                                                                                                        fully_controlled_directions,
                                                                                                        compliant_frame,
                                                                                                        controller_data.control_links[i]);
            tasks[controller_data.control_links[i]]->disableInternalOtg();
            tasks[controller_data.control_links[i]]->disableSingularityHandling();
            tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
            tasks[controller_data.control_links[i]]->setOriControlGains(350, 25, 0);

        } else if (controller_data.control_links[i] == "ra_end_effector" || controller_data.control_links[i] == "la_end_effector") {

            // std::string pos_control_link;
            // if (controller_data.control_links[i] == "ra_end_effector") {
            //     pos_control_link = "ra_link5";
            // } else {
            //     pos_control_link = "la_link5";
            // }

            // // Split task into two separate 3 DOF task 
            // tasks[controller_data.control_links[i] + "_pos"] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
            //                                                                                             pos_control_link,
            //                                                                                             fully_controlled_directions,
            //                                                                                             uncontrolled_directions,
            //                                                                                             compliant_frame,
            //                                                                                             controller_data.control_links[i] + "_pos");

            // tasks[controller_data.control_links[i] + "_pos"]->setSingularityHandlingBounds(5e-1, 5e0);
            // tasks[controller_data.control_links[i] + "_pos"]->handleAllSingularitiesAsType1(true);
            // tasks[controller_data.control_links[i] + "_pos"]->disableInternalOtg();
            // tasks[controller_data.control_links[i] + "_pos"]->setPosControlGains(350, 25, 0);  

            // tasks[controller_data.control_links[i] + "_ori"] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
            //                                                                                             controller_data.control_links[i],
            //                                                                                             uncontrolled_directions,
            //                                                                                             fully_controlled_directions,
            //                                                                                             compliant_frame,
            //                                                                                             controller_data.control_links[i] + "_ori");

            // tasks[controller_data.control_links[i] + "_ori"]->disableSingularityHandling();
            // tasks[controller_data.control_links[i] + "_ori"]->disableInternalOtg();
            // tasks[controller_data.control_links[i] + "_ori"]->setOriControlGains(350, 25, 0);                    

            // 6 DOF task 
            // compliant_frame.translation() = Vector3d(0, 0, 0.05);
            tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                        controller_data.control_links[i],
                                                                                                        compliant_frame,
                                                                                                        controller_data.control_links[i]);
            tasks[controller_data.control_links[i]]->disableInternalOtg();
            tasks[controller_data.control_links[i]]->setSingularityHandlingBounds(3e-3, 3e-2);
            // tasks[controller_data.control_links[i]]->disableSingularityHandling();
            // tasks[controller_data.control_links[i]]->setSingularityHandlingBounds(1e-2, 1e-1);
            tasks[controller_data.control_links[i]]->handleAllSingularitiesAsType1(true);  // need to test 
            // tasks[controller_data.control_links[i]]->setSingularityHandlingGains(100, 20, 20);
            tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
            tasks[controller_data.control_links[i]]->setPosControlGains(350, 25, 0);
            tasks[controller_data.control_links[i]]->setOriControlGains(350, 25, 0);

            // add more singularity damping here
            tasks[controller_data.control_links[i]]->setSingularityHandlingGains(0, 100, 200);

        } else {
            // 6 DOF task 
            tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                        controller_data.control_links[i],
                                                                                                        compliant_frame,
                                                                                                        controller_data.control_links[i]);
            tasks[controller_data.control_links[i]]->disableInternalOtg();
            tasks[controller_data.control_links[i]]->setSingularityHandlingBounds(3e-3, 3e-2);
            // tasks[controller_data.control_links[i]]->disableSingularityHandling();
            tasks[controller_data.control_links[i]]->handleAllSingularitiesAsType1(true);  // need to test 
            // tasks[controller_data.control_links[i]]->setSingularityHandlingGains(100, 20, 20);
            tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
            tasks[controller_data.control_links[i]]->setPosControlGains(350, 25, 0);
            tasks[controller_data.control_links[i]]->setOriControlGains(350, 25, 0);

            // add more singularity damping here
            tasks[controller_data.control_links[i]]->setSingularityHandlingGains(0, 100, 200);
        }
    }

    auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
    joint_task->disableInternalOtg();
    VectorXd q_desired = robot->q();
    // q_desired(9) = 2.0;
    // q_desired(15) = 2.0;
    // q_desired << 0, 0, 0, 0, 0, 0, 
    //             0, -0.1, -0.25, 0.5, -0.25, 0.1, 
    //             0, 0.1, -0.25, 0.5, -0.25, -0.1, 
    //             0, 0,
    //             -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
    //             -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
    //             0, 0;
	joint_task->setGains(350, 25, 0);
    joint_task->setDynamicDecouplingType(SaiPrimitives::DynamicDecouplingType::FULL_DYNAMIC_DECOUPLING);
	joint_task->setGoalPosition(q_desired);  
    nominal_posture = q_desired;

    // create robot controller
    std::vector<shared_ptr<SaiPrimitives::TemplateTask>> task_list;
    for (auto task_name : controller_data.control_links) {
        // if (task_name == "ra_end_effector" || task_name == "la_end_effector") {
            // task_list.push_back(tasks[task_name + "_ori"]);
            // task_list.push_back(tasks[task_name + "_pos"]);
        // } else {
            task_list.push_back(tasks[task_name]);
        // }
    }
    task_list.push_back(joint_task);
	auto robot_controller = std::make_unique<SaiPrimitives::RobotController>(robot, task_list);

    // low pass filter for each body part 
    int cutoff_freq = 100;
    int sampling_rate = 1000;
    std::map<std::string, SaiCommon::ButterworthFilter*> lpf_filters;
    for (auto it = optitrack_data.body_index_mapping.begin(); it != optitrack_data.body_index_mapping.end(); ++it) {
        std::string body_part_name = it->first;
        int index = it->second;

        // lpf_filters[body_part_name] = new SaiCommon::ButterworthFilter(cutoff_freq, sampling_rate);
        lpf_filters[body_part_name] = new SaiCommon::ButterworthFilter(0.4);
        // lpf_filters[body_part_name]->initializeFilter(Vector3d::Zero());
    }

    std::vector<Affine3d> optitrack_current_pose;
    std::vector<Vector6d> optitrack_current_velocity;
    for (auto it = optitrack_data.body_index_mapping.begin(); it != optitrack_data.body_index_mapping.end(); ++it) {
        optitrack_current_pose.push_back(Affine3d::Identity());
        optitrack_current_velocity.push_back(Vector6d::Zero());
    }

    // initialize
    int state = INIT;

    bool first_loop = true;
    const int n_calibration_samples = 1;  // N second of samples
    int n_samples = 0;
    VectorXd robot_control_torques = VectorXd::Zero(dof);
    VectorXd prev_control_torques = VectorXd::Zero(dof);
    int reset_robot = 0;

    redis_client.setInt(MULTI_RESET_CONTROLLER_KEY[ROBOT_ID], reset_robot);

    // setup read and write callbacks
    redis_client.addToReceiveGroup(MULTI_TORO_JOINT_ANGLES_KEY[ROBOT_ID], robot_q);
    redis_client.addToReceiveGroup(MULTI_TORO_JOINT_VELOCITIES_KEY[ROBOT_ID], robot_dq);
    redis_client.addToReceiveGroup(MULTI_RESET_CONTROLLER_KEY[ROBOT_ID], reset_robot);

    std::map<std::string, Vector3d> optitrack_data_current_position;
    std::map<std::string, VectorXd> optitrack_data_current_orientation;  // quaternion
    std::map<std::string, Matrix3d> optitrack_data_current_orientation_rot;  // rot mat
    std::map<std::string, Vector3d> optitrack_data_current_linear_velocity;
    std::map<std::string, VectorXd> optitrack_data_current_angular_velocity;
    for (auto it = optitrack_data.body_index_mapping.begin(); it != optitrack_data.body_index_mapping.end(); ++it) {
        std::string body_part_name = it->first;
        int index = it->second;
        optitrack_data_current_position[body_part_name] = Vector3d::Zero();
        optitrack_data_current_orientation[body_part_name] = VectorXd::Zero(4);
        optitrack_data_current_orientation_rot[body_part_name] = Matrix3d::Identity();
        optitrack_data_current_linear_velocity[body_part_name] = Vector3d::Zero();
        optitrack_data_current_angular_velocity[body_part_name] = Vector3d::Zero();

        redis_client.addToReceiveGroup(std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::pos", optitrack_data_current_position[body_part_name]);
        redis_client.addToReceiveGroup(std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::ori", optitrack_data_current_orientation[body_part_name]);

        redis_client.addToReceiveGroup(std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::lin_vel", optitrack_data_current_linear_velocity[body_part_name]);
        redis_client.addToReceiveGroup(std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::ang_vel", optitrack_data_current_angular_velocity[body_part_name]);
    }

    // redis_client.addToSendGroup(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[ROBOT_ID], robot_control_torques);

	// create a loop timer
    int controller_counter = 0;
    runloop = true;
	double control_freq = 2000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

    // Add these above while (runloop)
    VectorXd prev_momentum = VectorXd::Zero(dof);
    double prev_kinetic_energy = 0.0;
    std::vector<double> segment_times;
    double last_segment_time = 0.0;  // initialize this if you're using cooldown checks
    double cooldown_time = 0.8;   
    double cumulative_delta_ke = 0.0;
    double cumulative_delta_p  = 0.0;
    size_t sample_count        = 1; 
    size_t ground_count        = 1; 
    size_t ground_sample_count = 1;  

	while (runloop) {
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();
        

        // execute read callback
        redis_client.receiveAllFromGroup();
        for (const auto& [part, index] : optitrack_data.body_index_mapping) {
            std::string prefix = std::to_string(ROBOT_ID) + "::" + std::to_string(index);
        
            auto& pos = optitrack_data_current_position[part];
            auto& ori = optitrack_data_current_orientation[part];
            // auto& rot_ori = optitrack_data_current_orientation_rot[part];
            auto& lin_vel = optitrack_data_current_linear_velocity[part];
            auto& ang_vel = optitrack_data_current_angular_velocity[part];
        
            if (pos.hasNaN()) std::cerr << "[ERROR] NaN in pos for " << part << "\n";
            if (ori.size() != 4 || ori.hasNaN()) std::cerr << "[ERROR] Invalid orientation for " << part << ", size: " << ori.size() << "\n";
            if (lin_vel.hasNaN()) std::cerr << "[ERROR] NaN in lin_vel for " << part << "\n";
            if (ang_vel.hasNaN()) std::cerr << "[ERROR] NaN in ang_vel for " << part << "\n";
        }
        

        // update robot model
        robot->setQ(robot_q);
        robot->setDq(robot_dq);
        robot->updateModel();

        // Inside the loop, after robot->updateModel()
        MatrixXd Mq = robot->M();                    // Inertia matrix
        double kinetic_energy = 0.5 * robot_dq.transpose() * Mq * robot_dq;

        VectorXd momentum = Mq * robot_dq;           // Current momentum
        VectorXd delta_p = momentum - prev_momentum; // Change from previous frame
        double momentum_threshold = 0.8;
        
        if (delta_p.norm() > momentum_threshold && (time - last_segment_time) > cooldown_time) {
            segment_times.push_back(time);
            last_segment_time = time;
        // if (true) {
            // Get timestamp from Redis (as a string)
            std::string timestamp = redis_client.get("timestamp");
            std::ofstream log_file("/Users/rheamalhotra/Desktop/robotics/optitrack_dance_demo/synergies/momentum_changes.txt", std::ios::app);
            if (log_file.is_open()) {
                // log_file << timestamp << " " << delta_p.norm() << "\n";
                log_file << timestamp << "\n";
                log_file.close();
            }
        }        
        double delta_ke     = kinetic_energy - prev_kinetic_energy;
        double delta_p_norm = (momentum - prev_momentum).norm();

        // accumulate absolute changes
        cumulative_delta_ke += std::abs(delta_ke);
        cumulative_delta_p  += delta_p_norm;
        ++sample_count;

        // update “previous” for next iteration
        prev_kinetic_energy = kinetic_energy;
        prev_momentum       = momentum;

        // compute running averages (guard against divide‐by‐zero)
        double avg_delta_ke = (sample_count > 0)
        ? cumulative_delta_ke / sample_count
        : 0.0;
        double avg_delta_p  = (sample_count > 0)
        ? cumulative_delta_p  / sample_count
        : 0.0;

        // std::cout<<"average KE"<<avg_delta_ke<<std::endl;
        // std::cout<<"average P"<<avg_delta_p<<std::endl;



        // read the reset state 
        // int reset_robot = redis_client.getInt(MULTI_RESET_CONTROLLER_KEY[ROBOT_ID]);
        if (reset_robot) {
            std::cout << "Controller Reset\n";
            state = INIT;
            redis_client.setInt(MULTI_RESET_CONTROLLER_KEY[ROBOT_ID], 0);
            redis_client.setInt(RESET_SIM_KEY, 0);
            continue;
        }

        // // read optitrack input and store in optitrack struct 
        // if (state == CALIBRATION || state == TRACKING) {

        //     for (auto it = optitrack_data.body_index_mapping.begin(); it != optitrack_data.body_index_mapping.end(); ++it) {

        //         std::string body_part_name = it->first;
        //         int index = it->second;
        //         // Vector3d current_position = redis_client.getEigen(std::to_string(optitrack_data.human_ids[ROBOT_ID]) + "::" + std::to_string(index) + "::pos");
        //         // MatrixXd quaternion_matrix = redis_client.getEigen(std::to_string(optitrack_data.human_ids[ROBOT_ID]) + "::" + std::to_string(index) + "::ori");
        //         Vector3d current_position = redis_client.getEigen(std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::pos");
        //         MatrixXd quaternion_matrix = redis_client.getEigen(std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::ori");
        //         if (quaternion_matrix.size() != 4) {
        //             std::cerr << "Error: Quaternion retrieved from Redis does not have 4 elements." << std::endl;
        //             continue;
        //         }
        //         //  std::vector<double> quaternion(4);
        //         // for (int j = 0; j < 4; ++j)
        //         // {
        //         //     quaternion[j] = quaternion_matrix(j, 0); // Assuming the quaternion is stored as a 4x1 matrix
        //         // }

        //         // Create the affine transformation
        //         Eigen::Affine3d current_pose = Eigen::Affine3d::Identity();
        //         current_pose.translation() = current_position;
        //         current_pose.linear() = quaternionToRotationMatrix(quaternion_matrix);

        //         // Overwrite map for pose 
        //         optitrack_data.current_pose[body_part_name] = current_pose;

        //         // DEBUG only for visualization
        //         if (DEBUG) {
        //             redis_client.setEigen("opti::" + body_part_name + "::pos", R_optitrack_to_sai * current_pose.translation());
        //             redis_client.setEigen("opti::" + body_part_name + "::ori", R_optitrack_to_sai * current_pose.linear() * R_optitrack_to_sai.transpose());
        //         }

        //     }
        //     // // If needed, store in other vectors as   well
        //     //     if (i <= 3) {
        //     //         current_primary_poses.push_back(current_pose);
        //     //     } else {
        //     //         current_secondary_poses.push_back(current_pose);
        //     //     }
        // }

        // // print out
        // for (int i = 1; i < NUM_RIGID_BODIES; ++i) {
        //     std::cout << "Rigid body " << i << " position: " << current_link_poses[i].translation().transpose() << std::endl;
        //     std::cout << "Rigid body " << i << " orientation: " << current_link_poses[i].rotation() << std::endl;
        // }
        // perform robustness checks with the optitrack input 

        if (state == INIT) {
            // start robot at default configuration and hold the posture
            joint_task->setGoalPosition(nominal_posture); //THIS OMMGANSDFKANSDFKL
            N_prec.setIdentity();
            joint_task->updateTaskModel(N_prec);
            robot_control_torques = joint_task->computeTorques();

            if (joint_task->goalPositionReached(1e-2)) {
                if (redis_client.getInt(MULTI_USER_READY_KEY[ROBOT_ID]) == 1) {
                    state = CALIBRATION;
                    // state = TEST;
                    first_loop = true;
                    n_samples = 0;

                    // nominal_posture(9) = 2.0;
                    // nominal_posture(15) = 2.0;
                    // nominal_posture(23) = -1.5;
                    // nominal_posture(30) = -1.5;  // elbows
                    // joint_task->setGoalPosition(nominal_posture);
                    continue;
                }

                for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                    it->second->reInitializeTask();
                }   

                // populate the initial starting pose in SAI 
                for (int i = 0; i < controller_data.control_links.size(); ++i) {
                    std::string control_link_name = controller_data.control_links[i];
                    sim_body_data.starting_pose[control_link_name].translation() = robot->positionInWorld(control_link_name, controller_data.control_points[i]);
                    sim_body_data.starting_pose[control_link_name].linear() = robot->rotationInWorld(control_link_name);
                }         

            }
        } else if (state == CALIBRATION) {
            // gather N samples of the user starting position to "zero" the starting position of the human operator 
            //REGISTER KEY INPUT & calibrations
            // tp get starting x_o and R_o x and rotation matrix
            // take n samples and average
            // 1 for each rigid body
            //R transpose R is the delta

            // wait for user ready key input 
            bool user_ready = redis_client.getBool(MULTI_USER_READY_KEY[ROBOT_ID]);

            if (user_ready) {
                // recalibrate 
                std::vector<std::string> link_names;
                std::vector<Affine3d> link_poses;
                // for (auto it = optitrack_data.current_pose.begin(); it != optitrack_data.current_pose.end(); ++it) {
                //     link_names.push_back(it->first);
                //     link_poses.push_back(it->second);
                // }
                Affine3d tmp_pose;
                for (auto it = optitrack_data_current_position.begin(); it != optitrack_data_current_position.end(); ++it) {
                    link_names.push_back(it->first);
                    tmp_pose.translation() = optitrack_data_current_position[it->first];
                    tmp_pose.linear() = quaternionToRotationMatrix(optitrack_data_current_orientation[it->first]);
                    // tmp_pose.linear() = optitrack_data_current_orientation[it->first];
                    link_poses.push_back(tmp_pose);

                    lpf_filters[it->first]->initializeFilter(tmp_pose.translation());
                }
                human->calibratePose(link_names, link_poses, first_loop);
                if (first_loop) {
                    first_loop = false;
                }

                if (n_samples > n_calibration_samples) {
                    state = TRACKING;
                    // state = TEST;
                    n_samples = 0;

                    // publish the starting poses in optitrack frame
                    auto initial_poses = human->getMultiInitialPose(controller_data.control_links);
                    // for (int i = 0; i < initial_poses.size(); ++i) 
                        // std::cout << "Link: " << controller_data.control_links[i] << "\n";
                        // std::cout << "Starting optitrack position: \n" << initial_poses[i].translation().transpose() << "\n";
                        // std::cout << "Starting optitrack rotation: \n" << initial_poses[i].linear() << "\n";
                        // std::cout << "--- \n";  
                    // }

                    // // Retrieve the initial positions of the right and left end effectors (legs)
                    Eigen::Vector3d initial_rleg_pos = human->getInitialPose("RL_foot").translation();
                    Eigen::Vector3d initial_lleg_pos = human->getInitialPose("LL_foot").translation();

                    // // Calculate the ground level as the average of the z-coordinates of the initial leg positions
                    double ground_z = (initial_rleg_pos.z() + initial_lleg_pos.z()) / 2.0;
                    // Eigen::Vector3d ground_pos(0, 0, ground_z);

                    // // Set the ground key in Redis
                    redis_client.set(GROUND, std::to_string(ground_z));

                    for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                        it->second->reInitializeTask();
                    }       

                    // populate the initial starting pose in SAI 
                    for (int i = 0; i < controller_data.control_links.size(); ++i) {
                        std::string control_link_name = controller_data.control_links[i];
                        sim_body_data.starting_pose[control_link_name].translation() = robot->positionInWorld(control_link_name, controller_data.control_points[i]);
                        sim_body_data.starting_pose[control_link_name].linear() = robot->rotationInWorld(control_link_name);
                    }

                    continue;

                } else {
                    n_samples++;
                }
            }
            // from the end effector and initialized world you can get open sai x and open sai R
            // and from optitrack OS R to OT

            // this is wrong 
            // std::vector<Affine3d> OS_X0;
            // std::vector<Affine3d> 0S_R0;

            // OT_X_rel, OT_R_rel = human->relativePose;

        } else if (state == TRACKING) {
            std::string ground_str = redis_client.get(GROUND);
            double ground_z = std::stod(ground_str);

            std::string timestamp = redis_client.get("timestamp");
            Eigen::VectorXd q = robot->q();
            Eigen::VectorXd dq = robot->dq();
            std::ofstream file("robot_q.csv", std::ios::app); // <-- append!

            if (file.is_open()) {
                file << timestamp;
                for (int i = 0; i < q.size(); ++i) {
                    file << "," << std::fixed << std::setprecision(6) << q(i);
                }
                file << std::endl; // new line
                file.close();
            } else {
                std::cerr << "Failed to open file for writing!" << std::endl;
            }
            std::ofstream file2("robot_dq.csv", std::ios::app);

            if (file2.is_open()) {
                file2 << timestamp;
                for (int i = 0; i < q.size(); ++i) {
                    file2 << "," << std::fixed << std::setprecision(6) << q(i);
                }
                file2 << std::endl; // new line
                file2.close();
            } else {
                std::cerr << "Failed to open file for writing!" << std::endl;
            }



            robot_controller->updateControllerTaskModels();

            // manually set type 1 posture for all tasks to be the starting posture 
            // for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                // it->second->setType1Posture(nominal_posture);
            // }

            // // update simulation current pose 
            // std::vector<Affine3d> sim_current_pose;
            // for (int i = 0; i < controller_data.control_links.size(); ++i) {
            //     Affine3d current_pose = Affine3d::Identity();
            //     current_pose.translation() = robot->positionInWorld(controller_data.control_links[i], controller_data.control_points[i]);
            //     current_pose.linear() = robot->rotationInWorld(controller_data.control_links[i]);
            //     sim_current_pose.push_back(current_pose);
            // }

            // package optitrack poses for relative pose
            // std::vector<Affine3d> optitrack_current_pose;
            // if (controller_counter % 10 == 0) {
            Affine3d tmp_current_pose;
            Vector6d tmp_current_vel;
            for (int i = 0; i < controller_data.control_links.size(); ++i) {

                // push to lpdf
                // tmp_current_pose.translation() = lpf_filters[controller_data.control_links[i]]->update(optitrack_data_current_position[controller_data.control_links[i]]);

                tmp_current_pose.translation() = optitrack_data_current_position[controller_data.control_links[i]];
                tmp_current_pose.linear() = quaternionToRotationMatrix(optitrack_data_current_orientation[controller_data.control_links[i]]);
                optitrack_current_pose[i] = tmp_current_pose;

                // optitrack_current_pose.push_back(optitrack_data.current_pose[controller_data.control_links[i]]);
                // std::cout << "current position: " << tmp_current_pose.translation() << "\n";

                tmp_current_vel.head(3) = optitrack_data_current_linear_velocity[controller_data.control_links[i]];
                tmp_current_vel.tail(3) = optitrack_data_current_angular_velocity[controller_data.control_links[i]];
                optitrack_current_velocity[i] = tmp_current_vel;
            }
            // }

             // -------- set task goals and compute control torques
            robot_control_torques.setZero();

            auto relative_poses = human->relativePose(controller_data.control_links, optitrack_current_pose);  // returns back relative poses in order of control links 

            int i = 0;
            for (auto name : controller_data.control_links) {

                Vector3d desired_position = sim_body_data.starting_pose[name].translation() + MOTION_SF * relative_poses[i].translation();
                Matrix3d desired_orientation = relative_poses[i].linear() * sim_body_data.starting_pose[name].linear();
                // desired_orientation = reOrthogonalizeRotationMatrix(desired_orientation);
                
                // std::cout<<"GROUND: " <<ground_count << ", " <<ground_sample_count<<std::endl;
                double ground_ratio =
                    static_cast<double>(ground_count) 
                / static_cast<double>(ground_sample_count);

                // print with, e.g., 2 decimal places
                std::cout 
                << "GROUND ratio: " 
                << std::fixed 
                << std::setprecision(2) 
                << ground_ratio 
                << std::endl;
                
                if (desired_position.z() > ground_z)
                {
                    ++ground_count;
                }
                ++ground_sample_count;

                    tasks[name]->setGoalPosition(desired_position);
                    tasks[name]->setGoalOrientation(desired_orientation);
                    tasks[name]->setGoalLinearVelocity(optitrack_current_velocity[i].head(3));
                    tasks[name]->setGoalAngularVelocity(optitrack_current_velocity[i].tail(3));
                // }
                i++;
            }

            // for (auto name : controller_data.control_links) {
            //     robot_control_torques += tasks[name]->computeTorques();
            // }

            // std::cout << robot_controller->computeControlTorques().transpose() << "\n";

            robot_control_torques = robot_controller->computeControlTorques() + robot->coriolisForce();
            
            // std::cout << robot_control_torques.transpose() << "\n";

            // // saturate the absolute robot control torques
            // for (int i = 0; i < robot_control_torques.size(); ++i) {
            //     if (std::abs(robot_control_torques(i)) > MAX_JOINT_TORQUE) {
            //         robot_control_torques(i) *= MAX_JOINT_TORQUE / robot_control_torques(i);
            //     } 
            // }

            // // saturate the torque spike
            // for (int i = 6; i < robot_control_torques.size(); ++i) {
            //     if (std::abs(robot_control_torques(i) - prev_control_torques(i)) > MAX_TORQUE_SPIKE) {
            //         robot_control_torques(i) = prev_control_torques(i) + sign(robot_control_torques(i)) * MAX_TORQUE_SPIKE;
            //     }
            // }

            // end 

        } else if (state == TEST) {
            std::cout << "Test\n";
            // want to measure relative motion in optitrack frame
            robot_controller->updateControllerTaskModels();

            // // manual method
            // N_prec.setIdentity();
            // for (auto name : controller_data.control_links) {
            //     tasks[name]->updateTaskModel(N_prec);
            //     N_prec = tasks[name]->getTaskAndPreviousNullspace();
            // }

            // // manually set type 1 posture for all tasks to be the starting posture 
            // for (auto it = tasks.begin(); it != tasks.end(); ++it) {
            //     it->second->setType1Posture(q_init);
            // }
        
             // -------- set task goals and compute control torques
            robot_control_torques.setZero();

            // task motion
            // tasks["hip_base"]->setGoalPosition(sim_body_data.starting_pose["hip_base"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));
            // tasks["RL_foot"]->setGoalPosition(sim_body_data.starting_pose["RL_foot"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));
            // tasks["LL_foot"]->setGoalPosition(sim_body_data.starting_pose["LL_foot"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));

            tasks["ra_end_effector"]->setGoalPosition(sim_body_data.starting_pose["ra_end_effector"].translation() + 10.0 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));
            // tasks["la_end_effector"]->setGoalPosition(sim_body_data.starting_pose["la_end_effector"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));

            // for (auto name : controller_data.control_links) {
                // robot_control_torques += tasks[name]->computeTorques();
            // }

            robot_control_torques = robot_controller->computeControlTorques() + robot->coriolisForce();
            // std::cout << robot_control_torques.transpose() << "\n";
            // robot_control_torques.segment(6, 12).setZero();
            // robot_control_torques.segment(20, ).setZero();
        } else if (state == RESET) {
            // reset initializers for variables if needed

            state = INIT;
            continue;
        }

        if (isnan(robot_control_torques(0))) {
            // throw runtime_error("nan torques");
            std::cout << "nan torques: setting to zero torques\n";
            // robot_control_torques = prev_control_torques;
            robot_control_torques.setZero();
            // throw runtime_error("nan torques");
        }

        prev_control_torques = robot_control_torques;
        redis_client.setEigen(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[ROBOT_ID], robot_control_torques);
        controller_counter++;
        
	}

	timer.stop();
	cout << "\nControl loop timer stats:\n";
	timer.printInfoPostRun();
    redis_client.setEigen(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[ROBOT_ID], VectorXd::Zero(38));
	
}
