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
             std::shared_ptr<Human> human_syn,
             std::shared_ptr<SaiModel::SaiModel> robot,
             OptitrackData& motiongpt_data,
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
Matrix3d R_motiongpt_to_sai;
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
    auto motiongpt_data = OptitrackData();
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

    // // MOTION GPT MAPPING
    double deg2rad = M_PI / 180.0;
    double x_rot = 90.0 * deg2rad;
    double z_rot = 90.0 * deg2rad;

    // Rotation: first X, then Z (right-multiplied)
    R_motiongpt_to_sai =
        AngleAxisd(z_rot, Vector3d::UnitZ()) *
        AngleAxisd(x_rot, Vector3d::UnitX());

    // create map between body part names and indices 
    std::map<std::string, int> body_index_mapping; 
    for (int i = 0; i < body_part_names.size(); ++i) {
        body_index_mapping[body_part_names[i]] = optitrack_ids[i];
    }
    motiongpt_data.body_index_mapping = body_index_mapping;
    optitrack_data.body_index_mapping = body_index_mapping;    

    // create map between body part names and transformations
    for (int i = 0; i < body_part_names.size(); ++i) {
        optitrack_data.current_pose[body_part_names[i]] = Affine3d::Identity();
        motiongpt_data.current_pose[body_part_names[i]] = Affine3d::Identity();
    }

    optitrack_data.human_ids = human_ids;
    motiongpt_data.human_ids = human_ids;

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

    // Human that reads from MotionGPT-transformed keys
    auto human = std::make_shared<Optitrack::Human>(controller_data.control_links);
    human->setMultiRotationReference(controller_data.control_links, {controller_data.control_links.size(), R_motiongpt_to_sai});

    // Human that reads directly from OptiTrack keys
    auto human_syn = std::make_shared<Optitrack::Human>(controller_data.control_links);
    human_syn->setMultiRotationReference(controller_data.control_links, {controller_data.control_links.size(), R_optitrack_to_sai});

    // start controller thread
    thread control_thread(control, human, human_syn, robot, std::ref(motiongpt_data), std::ref(optitrack_data), std::ref(sim_body_data), std::ref(controller_data), nominal_posture);

    control_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void control(std::shared_ptr<Optitrack::Human> human,
             std::shared_ptr<Optitrack::Human> human_syn,
             std::shared_ptr<SaiModel::SaiModel> robot,
             OptitrackData& motiongpt_data,
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

    std::vector<Affine3d> motiongpt_current_pose;
    std::vector<Vector6d> motiongpt_current_velocity;
    for (auto it = motiongpt_data.body_index_mapping.begin(); it != motiongpt_data.body_index_mapping.end(); ++it) {
        motiongpt_current_pose.push_back(Affine3d::Identity());
        motiongpt_current_velocity.push_back(Vector6d::Zero());
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

    std::map<std::string, Vector3d> motiongpt_data_current_position;
    std::map<std::string, VectorXd> motiongpt_data_current_orientation;  // quaternion
    std::map<std::string, Matrix3d> motiongpt_data_current_orientation_rot;  // rot mat
    std::map<std::string, Vector3d> motiongpt_data_current_linear_velocity;
    std::map<std::string, VectorXd> motiongpt_data_current_angular_velocity;
    for (auto it = motiongpt_data.body_index_mapping.begin(); it != motiongpt_data.body_index_mapping.end(); ++it) {
        std::string body_part_name = it->first;
        int index = it->second;
        motiongpt_data_current_position[body_part_name] = Vector3d::Zero();
        motiongpt_data_current_orientation[body_part_name] = VectorXd::Zero(4);
        motiongpt_data_current_orientation_rot[body_part_name] = Matrix3d::Identity();
        motiongpt_data_current_linear_velocity[body_part_name] = Vector3d::Zero();
        motiongpt_data_current_angular_velocity[body_part_name] = Vector3d::Zero();

        redis_client.addToReceiveGroup("motiongpt::" + std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::pos", motiongpt_data_current_position[body_part_name]);
        redis_client.addToReceiveGroup("motiongpt::" + std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::ori", motiongpt_data_current_orientation[body_part_name]);

        redis_client.addToReceiveGroup("motiongpt::" + std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::lin_vel", motiongpt_data_current_linear_velocity[body_part_name]);
        redis_client.addToReceiveGroup("motiongpt::" + std::to_string(ROBOT_ID) + "::" + std::to_string(index) + "::ang_vel", motiongpt_data_current_angular_velocity[body_part_name]);
    }

    int controller_counter = 0;
    runloop = true;
	double control_freq = 2000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);


    VectorXd prev_momentum = VectorXd::Zero(dof);
    std::vector<double> segment_times;
    double last_segment_time = 0.0;  
    double cooldown_time = 0.2;     

	while (runloop) {
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();
        

        // execute read callback
        redis_client.receiveAllFromGroup();
        for (const auto& [part, index] : optitrack_data.body_index_mapping) {
            std::string prefix = std::to_string(ROBOT_ID) + "::" + std::to_string(index);
        
            auto& pos = optitrack_data_current_position[part];
            auto& ori = optitrack_data_current_orientation[part];
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

        auto joint_gravity = robot->jointGravityVector();
        double gTg = joint_gravity.transpose() * joint_gravity; // g^T g

        std::ofstream joint_file("/Users/rheamalhotra/Desktop/robotics/optitrack_dance_demo/synergies/joint_gravity.txt", std::ios::app);
        double kinetic_energy = 0.5 * robot_dq.transpose() * Mq * robot_dq;
        std::ofstream log_file("/Users/rheamalhotra/Desktop/robotics/optitrack_dance_demo/synergies/kinetic_energy.txt", std::ios::app);
        if (kinetic_energy && log_file.is_open() && joint_file.is_open()) {
            log_file << kinetic_energy;
            log_file << "\n"; 
            log_file.close();
            joint_file << gTg << "\n";
            joint_file.close();
        }

        VectorXd momentum = Mq * robot_dq;           // Current momentum
        VectorXd delta_p = momentum - prev_momentum; // Change from previous frame
        double momentum_threshold = 1.0;
        
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

        prev_momentum = momentum;


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
                // recalibrate motiongpt
                std::vector<std::string> motiongpt_link_names;
                std::vector<Affine3d> motiongpt_link_poses;
                Affine3d motiongpt_tmp_pose;
                for (auto it = motiongpt_data_current_position.begin(); it != motiongpt_data_current_position.end(); ++it) {
                    motiongpt_link_names.push_back(it->first);
                    motiongpt_tmp_pose.translation() = motiongpt_data_current_position[it->first];
                    motiongpt_tmp_pose.linear() = quaternionToRotationMatrix(motiongpt_data_current_orientation[it->first]);
                    motiongpt_link_poses.push_back(motiongpt_tmp_pose);

                    lpf_filters[it->first]->initializeFilter(motiongpt_tmp_pose.translation());
                }
                human->calibratePose(motiongpt_link_names, motiongpt_link_poses, first_loop);
                // calibrate optitrack
                std::vector<std::string> link_names;
                std::vector<Affine3d> link_poses;
                Affine3d tmp_pose;
                for (auto it = optitrack_data_current_position.begin(); it != optitrack_data_current_position.end(); ++it) {
                    link_names.push_back(it->first);
                    tmp_pose.translation() = optitrack_data_current_position[it->first];
                    tmp_pose.linear() = quaternionToRotationMatrix(optitrack_data_current_orientation[it->first]);
                    link_poses.push_back(tmp_pose);

                    lpf_filters[it->first]->initializeFilter(tmp_pose.translation());
                }
                human_syn->calibratePose(link_names, link_poses, first_loop);
                if (first_loop) {
                    first_loop = false;
                }

                if (n_samples > n_calibration_samples) {
                    state = TRACKING;
                    n_samples = 0;

                    // publish the starting poses in MOTIONGPT frame
                    auto initial_poses = human->getMultiInitialPose(controller_data.control_links);

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

        } else if (state == TRACKING) {
            // want to measure relative motion in optitrack frameR
            robot_controller->updateControllerTaskModels();

            // MOTIONGPT
            Affine3d motiongpt_tmp_current_pose;
            Vector6d motiongpt_tmp_current_vel;
            for (int i = 0; i < controller_data.control_links.size(); ++i) {

                motiongpt_tmp_current_pose.translation() = motiongpt_data_current_position[controller_data.control_links[i]];
                motiongpt_tmp_current_pose.linear() = quaternionToRotationMatrix(motiongpt_data_current_orientation[controller_data.control_links[i]]);
                motiongpt_current_pose[i] = motiongpt_tmp_current_pose;
                motiongpt_tmp_current_vel.head(3) = motiongpt_data_current_linear_velocity[controller_data.control_links[i]];
                motiongpt_tmp_current_vel.tail(3) = motiongpt_data_current_angular_velocity[controller_data.control_links[i]];
                motiongpt_current_velocity[i] = motiongpt_tmp_current_vel;
            }
            // OPTITRACK
            Affine3d optitrack_tmp_current_pose;
            Vector6d optitrack_tmp_current_vel;
            for (int i = 0; i < controller_data.control_links.size(); ++i) {

                optitrack_tmp_current_pose.translation() = optitrack_data_current_position[controller_data.control_links[i]];
                optitrack_tmp_current_pose.linear() = quaternionToRotationMatrix(optitrack_data_current_orientation[controller_data.control_links[i]]);
                optitrack_current_pose[i] = optitrack_tmp_current_pose;
                optitrack_tmp_current_vel.head(3) = optitrack_data_current_linear_velocity[controller_data.control_links[i]];
                optitrack_tmp_current_vel.tail(3) = optitrack_data_current_angular_velocity[controller_data.control_links[i]];
                optitrack_current_velocity[i] = optitrack_tmp_current_vel;
            }



             // -------- set task goals and compute control torques
            robot_control_torques.setZero();

            auto optitrack_relative_poses = human_syn->relativePose(controller_data.control_links, optitrack_current_pose);  
            auto motiongpt_relative_poses = human->relativePose(controller_data.control_links, motiongpt_current_pose); 
            
            double alpha = 0.5; // <-- set your interpolation weight between 0 and 1
            
            int i = 0;
            for (const auto& name : controller_data.control_links) {
                Vector3d interp_translation;
                Eigen::Matrix3d interp_rotation;

                if (i == 0) {
                // if (false) {
                    interp_translation = optitrack_relative_poses[i].translation();
                    interp_rotation = optitrack_relative_poses[i].linear();
                } else {
                    interp_translation = motiongpt_relative_poses[i].translation();
                    interp_rotation = motiongpt_relative_poses[i].linear();
                }

                Vector3d desired_position = sim_body_data.starting_pose[name].translation() + MOTION_SF * interp_translation;
                Matrix3d desired_orientation = interp_rotation * sim_body_data.starting_pose[name].linear();

                tasks[name]->setGoalPosition(desired_position);
                tasks[name]->setGoalOrientation(desired_orientation);
                tasks[name]->setGoalLinearVelocity(optitrack_current_velocity[i].head(3));
                tasks[name]->setGoalAngularVelocity(optitrack_current_velocity[i].tail(3));
                i++;
            }

            robot_control_torques = robot_controller->computeControlTorques() + robot->coriolisForce();


        } else if (state == TEST) {
            std::cout << "Test\n";
            // want to measure relative motion in optitrack frame
            robot_controller->updateControllerTaskModels();

             // -------- set task goals and compute control torques
            robot_control_torques.setZero();
            tasks["ra_end_effector"]->setGoalPosition(sim_body_data.starting_pose["ra_end_effector"].translation() + 10.0 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));

            robot_control_torques = robot_controller->computeControlTorques() + robot->coriolisForce();
        } else if (state == RESET) {
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

        // Add camera tracking
        string link_name = "neck_link2"; // head link
        Affine3d transform = robot->transformInWorld(link_name); // p_base = T * p_link
        MatrixXd rot = transform.rotation();
        VectorXd pos = transform.translation();
        VectorXd vert_axis = rot.col(2); // straight up from head (pos z)
        VectorXd lookat = rot.col(0); // straight ahead of head (pos x)
        
        VectorXd offset(3);
        offset << -2.8, 0.0, -1.1;
        pos += offset;

        controller_counter++;
        
	}

	timer.stop();
	cout << "\nControl loop timer stats:\n";
	timer.printInfoPostRun();
    redis_client.setEigen(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[ROBOT_ID], VectorXd::Zero(38));
	
}
