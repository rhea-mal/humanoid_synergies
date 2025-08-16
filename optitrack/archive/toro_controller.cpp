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

bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;
using namespace Optitrack;

// specify urdf and robots
const string robot_file = "./resources/model/HRP4c.urdf";
const string robot_name = "HRP4C0";
const string camera_name = "camera_fixed";

// globals
VectorXd nominal_posture;
VectorXd control_torques;
VectorXd ui_torques;
mutex mutex_update;
mutex mutex_torques;

const double MOTION_SF = 1.0;
const double MAX_RADIUS_ARM = 0.5;  // saturate arms within a sphere distance of the pelvis 

enum State {
    RESET = 0,
    CALIBRATION,
    TRACKING,
    TEST
};

// this is OT R transpose only
Eigen::Matrix3d quaternionToRotationMatrix(const VectorXd& quat)
{
    Eigen::Quaterniond q;
    q.x() = quat[0];
    q.y() = quat[1];
    q.z() = quat[2];
    q.w() = quat[3];
    return q.toRotationMatrix();
}

void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> simulation);

void control(std::shared_ptr<Human> human,
             std::shared_ptr<SaiModel::SaiModel> robot,
             const std::vector<int> indices,
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

Matrix3d R_camera_world = Matrix3d::Identity();
Matrix3d R_mirror;
Matrix3d R_optitrack_to_sai = (AngleAxisd(M_PI / 2, Vector3d::UnitZ()) * AngleAxisd(0 * M_PI / 2, Vector3d::UnitX())).toRotationMatrix();

int main(int argc, char** argv) {

    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // set initial z rotation viewing angle to be 0 (adjusted later if needed)
    redis_client.setDouble(Z_VIEWING_ANGLE, 180);
    // R_optitrack_to_sai *= AngleAxisd(90 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix();

    auto robot = make_shared<SaiModel::SaiModel>(robot_file, false);
    nominal_posture = robot->q();
    robot->updateModel();

    // load human class for motion tracking
    // std::vector<std::string> link_names = {"trunk", "la_link6", "ra_link6", \
    //                                         "neck_link2",};
    // std::vector<std::string> link_names = {"neck_link2", "LL_foot", "la_end_effector", "hip_base", "RL_foot", "ra_end_effector"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<std::string> link_names = {"neck_link2", "hip_base", "LL_foot", "RL_foot", "la_end_effector", "ra_end_effector"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<std::string> link_names = {"neck_link2", "trunk_rz", "hip_base", "LL_foot", "RL_foot", "la_end_effector", "ra_end_effector"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<std::string> link_names = {"neck_link2", "trunk_rz", "hip_base", "LL_foot", "RL_foot"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<std::string> link_names = {"neck_link2", "la_link4", "ra_link4", "hip_base", "LL_foot", "RL_foot", "la_end_effector", "ra_end_effector"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<int> indices = {5, 3, 2, 47, 50, 23, 32};  // head, chest | hip, hands, feet 

    // std::vector<std::string> link_names = {"neck_link2", "trunk_rz", "la_end_effector", "ra_end_effector", "la_link4", "ra_link4", "hip_base", "LL_foot", "RL_foot"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<int> indices = {5, 3, 23, 32, 8, 27, 2, 47, 50};

    // std::vector<std::string> link_names = {"neck_link2", "trunk_rz", "la_end_effector", "ra_end_effector", "hip_base", "LL_foot", "RL_foot"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    // std::vector<int> indices = {5, 3, 23, 32, 2, 47, 50};

    std::vector<std::string> link_names = {"neck_link2", "trunk_rz", "hip_base", "LL_foot", "RL_foot", "la_end_effector", "ra_end_effector"};  // MUST MATCH THE CONTROLLER LINK NAMES !!!
    std::vector<int> indices = {5, 3, 2, 47, 50, 23, 32};
    
    auto human = std::make_shared<Optitrack::Human>(link_names);
    // human->setMultiRotationReference(link_names, {R_optitrack_to_sai, R_optitrack_to_sai, R_optitrack_to_sai, R_optitrack_to_sai, R_optitrack_to_sai, R_optitrack_to_sai, R_optitrack_to_sai, R_optitrack_to_sai});
    human->setMultiRotationReference(link_names, {link_names.size(), R_optitrack_to_sai});
    // initialize containers and keys 
    control_torques = VectorXd::Zero(robot->dof());
    redis_client.setInt(USER_READY_KEY, 0);

    // start controller thread
    thread control_thread(control, human, robot, indices, nominal_posture);

    control_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void control(std::shared_ptr<Optitrack::Human> human,
             std::shared_ptr<SaiModel::SaiModel> robot,
             const std::vector<int> indices,
             const VectorXd& q_init) {

    // create redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // reset user ready key 
    redis_client.set(USER_READY_KEY, "0");

	// update robot model and initialize control vectors
    robot->setQ(redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
    robot->setDq(redis_client.getEigen(TORO_JOINT_VELOCITIES_KEY));
    robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);
   
    // create maps for tasks
    std::map<std::string, std::shared_ptr<SaiPrimitives::MotionForceTask>> primary_tasks;
    std::map<std::string, std::shared_ptr<SaiPrimitives::MotionForceTask>> secondary_tasks;

    // const std::vector<std::string> primary_control_links = {"LL_foot", "la_end_effector", "hip_base", "RL_foot", "ra_end_effector"};

    const std::vector<std::string> primary_control_links = {"hip_base", "LL_foot", "RL_foot", "la_end_effector", "ra_end_effector"};
    const std::vector<Vector3d> primary_control_points = {Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0),  Vector3d(0, 0, 0), Vector3d(0, 0, 0)};

    // const std::vector<std::string> primary_control_links = {"hip_base", "LL_foot", "RL_foot"};
    // const std::vector<Vector3d> primary_control_points = {Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};

    // const std::vector<std::string> primary_control_links = {"trunk_rz", "LL_foot", "RL_foot", "la_end_effector", "ra_end_effector"};
    // const std::vector<std::string> primary_control_links = {"hip_base", "LL_foot", "RL_foot"};
    // const std::vector<Vector3d> primary_control_points = {Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};

    // const std::vector<std::string> primary_control_links = {"hip_base", "la_end_effector", "ra_end_effector"};
    // const std::vector<Vector3d> primary_control_points = {Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};
    // const std::vector<std::string> secondary_control_links = {"neck_link2", "la_link4", "ra_link4", "LL_KOSY_L56", "RL_KOSY_L56"};
    // const std::vector<Vector3d> secondary_control_points = {Vector3d(0, 0, 0.11), Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};
    // const std::vector<std::string> secondary_control_links = {"neck_link2", "la_link4", "ra_link4"};
    // const std::vector<Vector3d> secondary_control_points = {Vector3d(0, 0, 0.11), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};
    // const std::vector<std::string> secondary_control_links = {"neck_link2", "trunk_rz", "la_end_effector", "ra_end_effector", "la_link4", "ra_link4"};
    // const std::vector<Vector3d> secondary_control_points = {Vector3d(0, 0, 0.11), Vector3d(0, 0, 0.15), Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};
    // const std::vector<std::string> secondary_control_links = {"neck_link2", "trunk_rz", "la_end_effector", "ra_end_effector"};
    // const std::vector<Vector3d> secondary_control_points = {Vector3d(0, 0, 0.11), Vector3d(0, 0, 0.15), Vector3d(0, 0, 0), Vector3d(0, 0, 0)};
    const std::vector<std::string> secondary_control_links = {"neck_link2", "trunk_rz"};
    const std::vector<Vector3d> secondary_control_points = {Vector3d(0, 0, 0.11), Vector3d(0, 0, 0.15)};

    // fill vectors for starting poses of the primary and secondary control links in the SAI world 
    std::vector<Affine3d> primary_links_initial_pose = {primary_control_links.size(), Affine3d::Identity()};
    std::vector<Affine3d> secondary_links_initial_pose = {secondary_control_links.size(), Affine3d::Identity()};

    std::vector<std::string> complete_link_names;
    for (auto name : secondary_control_links) {
        complete_link_names.push_back(name);
    }
    for (auto name : primary_control_links) {
        complete_link_names.push_back(name);
    }

    for (int i = 0; i < primary_control_links.size(); ++i) {        
        Affine3d compliant_frame = Affine3d::Identity();
        compliant_frame.translation() = primary_control_points[i];
        primary_tasks[primary_control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot, primary_control_links[i], compliant_frame, primary_control_links[i]);
        primary_tasks[primary_control_links[i]]->disableInternalOtg();
        primary_tasks[primary_control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
        primary_tasks[primary_control_links[i]]->handleAllSingularitiesAsType1(true);
        primary_tasks[primary_control_links[i]]->setPosControlGains(400, 40, 0);
        primary_tasks[primary_control_links[i]]->setOriControlGains(400, 40, 0);
        primary_tasks[primary_control_links[i]]->setSingularityHandlingBounds(7e-3, 7e-2);
    }

    std::vector<Vector3d> uncontrolled = {};
    std::vector<Vector3d> controlled = {Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ()};
    for (int i = 0; i < secondary_control_links.size(); ++i) {        
        Affine3d compliant_frame = Affine3d::Identity();
        compliant_frame.translation() = secondary_control_points[i];
        if (secondary_control_links[i] == "la_link4" || secondary_control_links[i] == "ra_link4") {
            secondary_tasks[secondary_control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot, secondary_control_links[i], compliant_frame, secondary_control_links[i]);        
        } else {
            secondary_tasks[secondary_control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot, secondary_control_links[i], controlled, uncontrolled, compliant_frame, secondary_control_links[i]);
        }
        secondary_tasks[secondary_control_links[i]]->disableInternalOtg();
        // secondary_tasks[secondary_control_links[i]]->disableSingularityHandling();  
        secondary_tasks[secondary_control_links[i]]->handleAllSingularitiesAsType1(true);
        // secondary_tasks[secondary_control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
        // secondary_tasks[secondary_control_links[i]]->setDynamicDecouplingType(SaiPrimitives::IMPEDANCE);
        secondary_tasks[secondary_control_links[i]]->setPosControlGains(100, 20, 0);
        secondary_tasks[secondary_control_links[i]]->setOriControlGains(100, 20, 0);
        secondary_tasks[secondary_control_links[i]]->setSingularityHandlingBounds(8e-3, 8e-2);

        if (secondary_control_links[i] != "la_end_effector" && secondary_control_links[i] != "ra_end_effector") {
            secondary_tasks[secondary_control_links[i]]->disableSingularityHandling();  
        }
    }

    // auto right_foot_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, "RL_end_effector", Affine3d::Identity());
    // right_foot_task->disableInternalOtg();
    // right_foot_task->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
    // right_foot_task->handleAllSingularitiesAsType1(true);
    // right_foot_task->setPosControlGains(100, 20, 0);
    // right_foot_task->setOriControlGains(100, 20, 0);

    // auto left_foot_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, "LL_end_effector", Affine3d::Identity());
    // left_foot_task->disableInternalOtg();
    // left_foot_task->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
    // left_foot_task->handleAllSingularitiesAsType1(true);
    // left_foot_task->setPosControlGains(100, 20, 0);
    // left_foot_task->setOriControlGains(100, 20, 0);

    auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
    joint_task->disableInternalOtg();
	VectorXd q_desired = robot->q();
	joint_task->setGains(100, 20, 0);
    joint_task->setDynamicDecouplingType(SaiPrimitives::DynamicDecouplingType::FULL_DYNAMIC_DECOUPLING);
	joint_task->setGoalPosition(q_desired);  

    // create robot controller
    std::vector<shared_ptr<SaiPrimitives::TemplateTask>> task_list;
    for (auto task_name : primary_control_links) {
        task_list.push_back(primary_tasks[task_name]);
    }
    // for (auto task_name : secondary_control_links) {
    //     task_list.push_back(secondary_tasks[task_name]);
    // }
    task_list.push_back(joint_task);
	auto robot_controller =
		make_unique<SaiPrimitives::RobotController>(robot, task_list);

    // initialize
    int state = CALIBRATION;

    bool first_loop = true;
    const int n_calibration_samples = 1 * 1000;  // N second of samples
    int n_samples = 0;
    VectorXd robot_control_torques = VectorXd::Zero(dof);

	// get starting poses in open sai world 
    std::vector<Affine3d> primary_starting_pose;
    std::vector<Affine3d> secondary_starting_pose;
    for (int i = 0; i < primary_control_links.size(); ++i) {
        Affine3d current_pose;
        current_pose.translation() = robot->position(primary_control_links[i], primary_control_points[i]);
        current_pose.linear() = robot->rotation(primary_control_links[i]);
        primary_starting_pose.push_back(current_pose);
    }
    for (int i = 0; i < secondary_control_links.size(); ++i) {
        Affine3d current_pose;
        current_pose.translation() = robot->position(secondary_control_links[i], secondary_control_points[i]);
        current_pose.linear() = robot->rotation(secondary_control_links[i]);
        secondary_starting_pose.push_back(current_pose);
    }

	// create a loop timer
    runloop = true;
	double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (runloop) {
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();

        // update robot model
        robot->setQ(redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
        robot->setDq(redis_client.getEigen(TORO_JOINT_VELOCITIES_KEY));
        robot->updateModel();

        // // receive optitrack input 
        //auto current_position = redis_client.getEigen(OPTITRACK_POS_KEY);
        //auto current_orientation = redis_client.getEigen(OPTITRACK_ORI_KEY);
        //const std::string OPTITRACK_POS_KEY = "Sai::optitrack::rigid_body_pos::";
        //const std::string OPTITRACK_ORI_KEY = "Sai::optitrack::rigid_body_ori::";
        // const int NUM_RIGID_BODIES = 6 + 2 - 2 + 1; 
        const int NUM_RIGID_BODIES = primary_control_links.size() + secondary_control_links.size();

        // parse input into usable format 
        std::vector<Affine3d> current_link_poses(NUM_RIGID_BODIES);
        std::vector<Affine3d> current_primary_poses = {};
        std::vector<Affine3d> current_secondary_poses = {};

        // // Mapping the indices from the dictionary (Python-like) to a C++ std::map
        // std::map<int, int> index_mapping = {
        //     {1, 5}, 
        //     {2, 50}, 
        //     {3, 23}, 
        //     {4, 2},  
        //     {5, 47}, 
        //     {6, 11}
        // };

        // // Mapping the indices from the dictionary (Python-like) to a C++ std::map
        // std::map<int, int> index_mapping = {
        //     {1, 5}, 
        //     {2, 47}, 
        //     {3, 23}, 
        //     {4, 2},  
        //     {5, 50}, 
        //     {6, 11}
        // };

        // // correct mapping 
        // std::map<int, int> index_mapping = {
        //     {1, 5}, // head; correct
        //     {2, 47}, // left foot; flipped
        //     {3, 23}, // left hand; correct
        //     {4, 2}, // torso; correct 
        //     {5, 50}, // right foot; flipped
        //     {6, 32}  // right hand 
        // };

        // // correct mapping 
        // std::map<int, int> index_mapping = {
        //     {1, 5}, // head; correct
        //     // {2, 2}, // torso; correct 
        //     {2, 3}, // torso; correct 
        //     {3, 47}, // left foot; flipped
        //     {4, 50}, // right foot; flipped
        //     {5, 23}, // left hand; correct
        //     {6, 32}  // right hand 
        // };


        // std::vector<int> indices = {5, 3, 2, 47, 50, 23, 32};  // head, chest | hip, hands, feet 
        // std::vector<int> indices = {5, 3, 2, 47, 50};  // head, chest | hip, hands, feet 
        // std::vector<int> indices = {5, 3, 47, 50, 23, 32};
        // std::vector<int> indices = {5, 8, 27, 2, 47, 50, 23, 32};

        // read optitrack input 
        if (state == CALIBRATION || state == TRACKING) {
            // for (int i = 0; i < NUM_RIGID_BODIES; ++i) {
            for (int i = 0; i < indices.size(); ++i) {
                // int index = index_mapping[i + 1];
                int index = indices[i];
        
                Eigen::Vector3d current_position = redis_client.getEigen("3::" + std::to_string(index) + "::pos");
                Eigen::MatrixXd quaternion_matrix = redis_client.getEigen("3::" + std::to_string(index) + "::ori");

                if (quaternion_matrix.size() != 4)
                {
                    std::cerr << "Error: Quaternion retrieved from Redis does not have 4 elements." << std::endl;
                    continue;
                }
                //  std::vector<double> quaternion(4);
                // for (int j = 0; j < 4; ++j)
                // {
                //     quaternion[j] = quaternion_matrix(j, 0); // Assuming the quaternion is stored as a 4x1 matrix
                // }

                // Create the affine transformation
                Eigen::Affine3d current_pose = Eigen::Affine3d::Identity();
                current_pose.translation() = current_position;
                current_pose.linear() = quaternionToRotationMatrix(quaternion_matrix);

                current_link_poses[i] = current_pose;

                if (i >= secondary_control_links.size()) {
                    current_primary_poses.push_back(current_pose);
                } else {
                    current_secondary_poses.push_back(current_pose);
                }
            // // If needed, store in other vectors as well
            //     if (i <= 3) {
            //         current_primary_poses.push_back(current_pose);
            //     } else {
            //         current_secondary_poses.push_back(current_pose);
            //     }
            }
        }

        // // print out
        // for (int i = 1; i < NUM_RIGID_BODIES; ++i) {
        //     std::cout << "Rigid body " << i << " position: " << current_link_poses[i].translation().transpose() << std::endl;
        //     std::cout << "Rigid body " << i << " orientation: " << current_link_poses[i].rotation() << std::endl;
        // }
        // perform robustness checks with the optitrack input 

        if (state == RESET) {
            // start robot at default configuration and hold the posture
            joint_task->setGoalPosition(nominal_posture);
            N_prec.setIdentity();
            joint_task->updateTaskModel(N_prec);
            robot_control_torques = joint_task->computeTorques();

            if (joint_task->goalPositionReached(1e-2)) {
                if (redis_client.getInt(USER_READY_KEY) == 1) {
                    state = CALIBRATION;
                    // state = TEST;
                    first_loop = true;
                    n_samples = 0;
                    continue;
                }
                
                for (auto it = primary_tasks.begin(); it != primary_tasks.end(); ++it) {
                    it->second->reInitializeTask();
                }

                for (auto it = secondary_tasks.begin(); it != secondary_tasks.end(); ++it) {
                    it->second->reInitializeTask();
                }

                // get robot link initial pose
                for (int i = 0; i < primary_control_links.size(); ++i) {
                    Affine3d link_pose = Affine3d::Identity();
                    link_pose.translation() = robot->positionInWorld(primary_control_links[i], primary_control_points[i]);
                    link_pose.linear() = robot->rotationInWorld(primary_control_links[i]);
                    primary_links_initial_pose[i] = link_pose;
                }

                for (int i = 0; i < secondary_control_links.size(); ++i) {
                    Affine3d link_pose = Affine3d::Identity();
                    link_pose.translation() = robot->positionInWorld(secondary_control_links[i], secondary_control_points[i]);
                    link_pose.linear() = robot->rotationInWorld(secondary_control_links[i]);
                    secondary_links_initial_pose[i] = link_pose;
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
            bool user_ready = redis_client.getBool(USER_READY_KEY);

            if (user_ready) {
                // recalibrate 
                human->calibratePose(complete_link_names, current_link_poses, n_calibration_samples, first_loop);
                if (first_loop) {
                    first_loop = false;
                }

                if (n_samples > n_calibration_samples) {
                    state = TRACKING;
                    n_samples = 0;

                    // publish the starting poses in optitrack frame
                    auto initial_poses = human->getMultiInitialPose(complete_link_names);
                    for (int i = 0; i < initial_poses.size(); ++i) {
                        std::cout << "Link: " << complete_link_names[i] << "\n";
                        std::cout << "Starting optitrack position: \n" << initial_poses[i].translation().transpose() << "\n";
                        std::cout << "Starting optitrack rotation: \n" << initial_poses[i].linear() << "\n";
                        std::cout << "--- \n";  
                    }

                    // Retrieve the initial positions of the right and left end effectors (legs)
                    Eigen::Vector3d initial_rleg_pos = human->getInitialPose("RL_foot").translation();
                    Eigen::Vector3d initial_lleg_pos = human->getInitialPose("LL_foot").translation();

                    // Calculate the ground level as the average of the z-coordinates of the initial leg positions
                    double ground_z = (initial_rleg_pos.z() + initial_lleg_pos.z()) / 2.0;
                    Eigen::Vector3d ground_pos(0, 0, ground_z);

                    // Set the ground key in Redis
                    redis_client.setEigen(GROUND, ground_pos);

                    // get robot link initial pose
                    for (int i = 0; i < primary_control_links.size(); ++i) {
                        Affine3d link_pose = Affine3d::Identity();
                        link_pose.translation() = robot->positionInWorld(primary_control_links[i], primary_control_points[i]);
                        link_pose.linear() = robot->rotationInWorld(primary_control_links[i]);
                        primary_links_initial_pose[i] = link_pose;
                    }

                    for (int i = 0; i < secondary_control_links.size(); ++i) {
                        Affine3d link_pose = Affine3d::Identity();
                        link_pose.translation() = robot->positionInWorld(secondary_control_links[i], secondary_control_points[i]);
                        link_pose.linear() = robot->rotationInWorld(secondary_control_links[i]);
                        secondary_links_initial_pose[i] = link_pose;
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
            // want to measure relative motion in optitrack frame
            robot_controller->updateControllerTaskModels();

            // manually set type 1 posture for all tasks to be the starting posture 
            for (auto it = primary_tasks.begin(); it != primary_tasks.end(); ++it) {
                    it->second->setType1Posture(q_init);
            }
            
            // manually set type 1 posture for all tasks to be the starting posture 
            for (auto it = secondary_tasks.begin(); it != secondary_tasks.end(); ++it) {
                    it->second->setType1Posture(q_init);
            }

            // c current in optitrack and rotation relative of optitrack
            // update model
            // N_prec.setIdentity();

            // // update right and left foot task model
            // right_foot_task->updateTaskModel(N_prec);
            // left_foot_task->updateTaskModel(N_prec);

            // // get task jacobian
            // MatrixXd J_feet_task(6 * 2, robot->dof());
            // J_feet_task.topRows(6) = robot->J("RL_end_effector", Vector3d::Zero());
            // J_feet_task.bottomRows(6) = robot->J("LL_end_effector", Vector3d::Zero());
            
            // N_prec = robot->nullspaceMatrix(J_feet_task);

            // update primary task model
            // for (auto it = primary_tasks.begin(); it != primary_tasks.end(); ++it) {
                // it->second->updateTaskModel(N_prec);
                // N_prec = it->second->getTaskAndPreviousNullspace();
            // }

            // // get primary task jacobian stack 
            // MatrixXd J_primary_tasks(6 * primary_control_links.size(), robot->dof());
            // for (int i = 0; i < primary_control_links.size(); ++i) {
            //     J_primary_tasks.block(6 * i, 0, 6, robot->dof()) = robot->J(primary_control_links[i], primary_control_points[i]);
            // }        
            // N_prec = robot->nullspaceMatrix(J_primary_tasks) * N_prec;

            // update secondary task model
            // for (auto it = secondary_tasks.begin(); it != secondary_tasks.end(); ++it) {
                // it->second->updateTaskModel(N_prec);
                // N_prec = it->second->getTaskAndPreviousNullspace();  
            // }

            // // get secondary task jacobian stack 
            // MatrixXd J_secondary_tasks(6 * secondary_control_links.size(), dof);
            // for (int i = 0; i < secondary_control_links.size(); ++i) {
            //     J_secondary_tasks.block(6 * i, 0, 6, dof) = robot->J(secondary_control_links[i], secondary_control_points[i]);
            // }
            // N_prec = robot->nullspaceMatrix(J_secondary_tasks) * N_prec;
                
            // redundancy completion
            // joint_task->updateTaskModel(N_prec);

            // -------- set task goals and compute control torques
            robot_control_torques.setZero();
            // std::cout << current_primary_poses[0].linear() << "\n";
            auto primary_relative_poses = human->relativePose(primary_control_links, current_primary_poses);  // in order of instantiated order

            int i = 0;
            // primary task rotating from optitrack to the world
            //zero the optitrack position (when u press a button, average over a few seconds of optitrack reads as reference zero position for all motions in optitrack)
            // all motions are relative to initialized zero position as delta motions
            // press another button to reset world (re zero button ) IMPLEMENT THIS PYTHON KEYBOARD PRESS A KEY -> CHECK THAT IN HUMAN CLASS
            // read in optitrack frame data, rigid body dynamic object for each rigid body for each optitrack
            // play back each position and orientation
            // goal position and goal orientation (hands and feet and torso) from sai
            // apply similarity transform to rotate matrix to rotate 3d matrix
            // orientation in sai = optitrack times delta orientation in optitrack times rotation matrix transpose
            // secondary tasks is head
            // human will give you delta motion 
            for (auto name : primary_control_links) {
                // std::cout << "primary relative translation: \n" << primary_relative_poses[i].translation() << "\n";
                // std::cout << "primary relative rotation: \n" << primary_relative_poses[i].linear() << "\n";
                // std::cout << "starting pose: \n" << primary_links_initial_pose[i].translation().transpose() << "\n";
                // std::cout << "starting pose: \n" << primary_links_initial_pose[i].linear() << "\n";

                Vector3d desired_position = primary_links_initial_pose[i].translation() + MOTION_SF * primary_relative_poses[i].translation();
                // Vector3d chest_position = robot->positionInWorld("trunk_rz", Vector3d::Zero());
                // if ((desired_position - chest_position).norm() > MAX_RADIUS_ARM && (name == "la_end_effector" || name == "ra_end_effector")) {
                    // desired_position = (desired_position / desired_position.norm()) * MAX_RADIUS_ARM;
                // }

                // desired_position = R_optitrack_to_sai * current_primary_poses[i].translation();

                primary_tasks[name]->setGoalPosition(desired_position);
                primary_tasks[name]->setGoalOrientation(primary_relative_poses[i].linear() * primary_links_initial_pose[i].linear());

                // global coordinates
                // primary_tasks[name]->setGoalPosition(R_optitrack_to_sai * current_primary_poses[i].translation());
                // primary_tasks[name]->setGoalOrientation(current_primary_poses[i].linear());
                
                // redis_client.setEigen(TEST_POS, primary_links_initial_pose[1].translation() + MOTION_SF * primary_relative_poses[1].translation());
                // redis_client.setEigen(TEST_ORI, primary_relative_poses[1].linear() * primary_links_initial_pose[1].linear());
                
                // robot_control_torques += 1 * primary_tasks[name]->computeTorques();
                i++;
            }
            
            auto secondary_relative_poses = human->relativePose(secondary_control_links, current_secondary_poses);  
            i = 0;
            for (auto name : secondary_control_links) {
                // std::cout << "secondary relative translation: \n" << secondary_relative_poses[i].translation() << "\n";
                // std::cout << "secondary relative rotation: \n" << secondary_relative_poses[i].linear() << "\n";
                secondary_tasks[name]->setGoalPosition(secondary_links_initial_pose[i].translation() + MOTION_SF * secondary_relative_poses[i].translation());
                secondary_tasks[name]->setGoalOrientation(secondary_relative_poses[i].linear() * secondary_links_initial_pose[i].linear());
                // robot_control_torques += 1 * secondary_tasks[name]->computeTorques();

                // global coordinates
                // secondary_tasks[name]->setGoalPosition(R_optitrack_to_sai * current_secondary_poses[i].translation());
                // secondary_tasks[name]->setGoalOrientation(current_secondary_poses[i].linear());

                ++i;
            }

            // robot_control_torques += robot->coriolisForce() + joint_task->computeTorques();

            // robot_control_torques *= 0;
            robot_control_torques = robot_controller->computeControlTorques() + robot->coriolisForce();

            // robot_control_torques += right_foot_task->computeTorques() + left_foot_task->computeTorques();

        } else if (state == TEST) {
            // update model
            N_prec.setIdentity();

            // update primary task model
            for (auto it = primary_tasks.begin(); it != primary_tasks.end(); ++it) {
                it->second->updateTaskModel(N_prec);
                // N_prec = it->second->getTaskAndPreviousNullspace();
            }

            // get primary task jacobian stack 
            MatrixXd J_primary_tasks(6 * primary_control_links.size(), robot->dof());
            for (int i = 0; i < primary_control_links.size(); ++i) {
                J_primary_tasks.block(6 * i, 0, 6, robot->dof()) = robot->J(primary_control_links[i], primary_control_points[i]);
            }        
            N_prec = robot->nullspaceMatrix(J_primary_tasks) * N_prec;

            // update secondary task model
            for (auto it = secondary_tasks.begin(); it != secondary_tasks.end(); ++it) {
                it->second->updateTaskModel(N_prec);
                // N_prec = it->second->getTaskAndPreviousNullspace();  
            }

            // get secondary task jacobian stack 
            MatrixXd J_secondary_tasks(6 * secondary_control_links.size(), dof);
            for (int i = 0; i < secondary_control_links.size(); ++i) {
                J_secondary_tasks.block(6 * i, 0, 6, dof) = robot->J(secondary_control_links[i], secondary_control_points[i]);
            }
            N_prec = robot->nullspaceMatrix(J_secondary_tasks) * N_prec;
                
            // redundancy completion
            joint_task->updateTaskModel(N_prec);

            // -------- set task goals and compute control torques
            robot_control_torques.setZero();

            int i = 0;
            for (auto name : primary_control_links) {
                if (i != 0) {
                    // primary_tasks[name]->setGoalPosition(primary_starting_pose[i].translation() + 0.05 * Vector3d(sin(2 * M_PI * time), 0, sin(2 * M_PI * time)));
                    primary_tasks[name]->setGoalPosition(primary_starting_pose[i].translation() + \
                            1 * Vector3d(0.5 * sin(2 * M_PI * 0.1 * time), 0.5 * sin(2 * M_PI * 0.1 * time), 0.5 * sin(2 * M_PI * 0.1 * time)));
                    // primary_tasks[name]->setGoalOrientation();
                }
                robot_control_torques += primary_tasks[name]->computeTorques();
                ++i;
            }
            
            i = 0;
            for (auto name : secondary_control_links) {
                // secondary_tasks[name]->setGoalPosition();
                // secondary_tasks[name]->setGoalOrientation();
                robot_control_torques += secondary_tasks[name]->computeTorques();
                ++i;
            }

            robot_control_torques += 1 * joint_task->computeTorques() + robot->coriolisForce();  // compute joint task torques if DOF isn't filled

            robot_control_torques *= 0;
            robot_control_torques = robot_controller->computeControlTorques();
        }

		//------ compute the final torques
		{
			lock_guard<mutex> lock(mutex_torques);
			control_torques = robot_control_torques;
		}

        if (isnan(control_torques(0))) {
            throw runtime_error("nan torques");
        }

        redis_client.setEigen(TORO_JOINT_TORQUES_COMMANDED_KEY, control_torques);

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
        
        redis_client.setEigen(HEAD_POS, pos);
        redis_client.setEigen(HEAD_VERT_AXIS, vert_axis);
        redis_client.setEigen(HEAD_LOOK_AT, lookat + pos); // Assuming look-at point is in world frame, not relative to camera pos



	}
	timer.stop();
	cout << "\nControl loop timer stats:\n";
	timer.printInfoPostRun();
    redis_client.setEigen(TORO_JOINT_TORQUES_COMMANDED_KEY, 0 * control_torques);





	
}


// //------------------------------------------------------------------------------

// vector<Matrix3d> computeOrientations(const vector<string> &ori_links) {
//     vector<Matrix3d> R_list;
//     Vector3d oriVec;
//     Vector3d frontVec{1, 0, 0};
//     for (int i = 0; i < ori_links.size(); ++i) {
//         Vector3d upVec{0, 0, 1};
//         auto R_joint = all_ori[idx_map[ori_links[i]]];
//         R_list.push_back(R_joint*default_ori);
//     }
//     return R_list;
// }

// //------------------------------------------------------------------------------

// VectorXd computeTorques(shared_ptr<SaiModel::SaiModel> robot, const vector<string> &primary_links,
//     const vector<Vector3d> &primary_pos, const vector<string> &secondary_links,
//     const vector<Vector3d> &secondary_pos, const vector<Vector3d> &primary_des_pos,
//     const vector<Vector3d> &secondary_des_pos, const VectorXd &q_des,
//     const vector<string> &ori_links,
//     const vector<int> &primary_conf, const vector<int> &secondary_conf) {
//     // Necessary orientation task to keep feet straight
//     int n_links = ori_links.size();
//     MatrixXd J_ori_stack(3 * n_links, robot->dof());
//     MatrixXd Jw_tmp(3, robot->dof());
//     vector<Matrix3d> desired_orientation(n_links, Matrix3d::Identity());
//     desired_orientation[0] << 0, -1, 0, 1, 0, 0, 0, 0, 1;
//     for (int i = 1; i < n_links; ++i) {
//         desired_orientation[i] = desired_orientation[0];
//     }
//     desired_orientation = computeOrientations(ori_links);
    
//     VectorXd ori_error_stack(3 * n_links), w_stack(3 * n_links);
//     for (int i = 0; i < n_links; ++i) {
//         Jw_tmp = robot->Jw(ori_links[i]);
//         J_ori_stack.block(3 * i, 0, 3, robot->dof()) = Jw_tmp;
//         Vector3d w = Jw_tmp * robot->dq();
//         w_stack.segment(3 * i, 3) = w;
//         Matrix3d curr_ori = robot->rotation(ori_links[i]);

//         // scale down oritation jumps and reject it if too large
//         Eigen::AngleAxisd ori_diff(desired_orientation[i] * curr_ori.transpose());
//         if (abs(ori_diff.angle()) > ori_tol && abs(ori_diff.angle()) < ori_reject_thres) {
//             ori_diff.angle() = ori_tol;
//             desired_orientation[i] = ori_diff.toRotationMatrix() * curr_ori;
//         } else if (abs(ori_diff.angle()) >= ori_reject_thres) {
//             desired_orientation[i] = curr_ori;
//         }
//         Vector3d ori_error = SaiModel::orientationError(desired_orientation[i], curr_ori);
//         ori_error_stack.segment(3 * i, 3) = ori_error;
//     }
//     MatrixXd Lambda_ori, N_ori;
//     lambdaSmoothing(Lambda_ori, N_ori, robot, J_ori_stack, 1e-8, 1e-9);
//     VectorXd F_ori = -25 * ori_error_stack - 10 * w_stack;

//     // Primary task
//     n_links = n_primary_links;
//     MatrixXd J_primary_stack(3 * n_links, robot->dof());
//     MatrixXd Jv_tmp(3, robot->dof());
//     Vector3d x_tmp, v_tmp;
//     VectorXd x_primary(3 * n_links);
//     VectorXd v_primary(3 * n_links);
//     VectorXd x_primary_stack(3 * n_links);

//     VectorXd x_primary_conf(3 * n_links);

//     for (int i = 0; i < n_links; ++i) {
//         Jv_tmp = robot->Jv(primary_links[i], primary_pos[i]);
//         J_primary_stack.block(3 * i, 0, 3, robot->dof()) = Jv_tmp;
//         x_tmp = robot->position(primary_links[i], primary_pos[i]);
//         x_primary.segment(3 * i, 3) = x_tmp;
//         x_primary_stack.segment(3 * i, 3) = primary_des_pos[i];
//         v_tmp = robot->linearVelocity(primary_links[i], primary_pos[i]);
//         v_primary.segment(3 * i, 3) = v_tmp;
//         x_primary_conf.segment(3 * i, 3) << primary_conf[i] - 1, primary_conf[i] - 1, primary_conf[i] - 1;
//     }

//     MatrixXd Lambda_primary, N_primary;
//     lambdaSmoothing(Lambda_primary, N_primary, robot, J_primary_stack, 1e-1, 1e-2);

//     // Secondary task
//     n_links = n_secondary_links;
//     MatrixXd J_secondary_stack(3 * n_links, robot->dof());
//     VectorXd x_secondary(3 * n_links);
//     VectorXd v_secondary(3 * n_links);
//     VectorXd x_secondary_stack(3 * n_links);

//     VectorXd x_secondary_conf(3 * n_links);

//     for (int i = 0; i < n_links; ++i) {
//         Jv_tmp = robot->Jv(secondary_links[i], secondary_pos[i]);
//         J_secondary_stack.block(3 * i, 0, 3, robot->dof()) = Jv_tmp;
//         x_tmp = robot->position(secondary_links[i], secondary_pos[i]);
//         x_secondary.segment(3 * i, 3) = x_tmp;
//         x_secondary_stack.segment(3 * i, 3) = secondary_des_pos[i];
//         v_tmp = robot->linearVelocity(secondary_links[i], secondary_pos[i]);
//         v_secondary.segment(3 * i, 3) = v_tmp;
//         x_secondary_conf.segment(3 * i, 3) << secondary_conf[i] - 1, secondary_conf[i] - 1, secondary_conf[i] - 1;
//     }
//     MatrixXd Lambda_secondary, N_secondary;
//     lambdaSmoothing(Lambda_secondary, N_secondary, robot, J_secondary_stack * N_primary, 1e-1, 1e-2);

//     // Compute task forces
//     VectorXd F_primary = -15 * x_primary_conf.cwiseProduct(x_primary - x_primary_stack) - 5 * x_primary_conf.cwiseProduct(v_primary);
//     VectorXd F_secondary = -4 * x_secondary_conf.cwiseProduct(x_secondary - x_secondary_stack) - 1.5 * x_secondary_conf.cwiseProduct(v_secondary);
//     VectorXd tau = -0 * (robot->q() - q_des) - 4.0 * robot->dq();

//     // Compute torques
//     // return J_primary_stack.transpose() * Lambda_primary * F_primary
//     //             + 1 * (J_secondary_stack * N_primary).transpose() * Lambda_secondary * F_secondary
//     //             + 1 * (MatrixXd::Identity(robot->dof(), robot->dof()) * N_primary * N_secondary).transpose() * robot->_M * tau;
//     return J_ori_stack.transpose() * Lambda_ori * F_ori +
//            (J_primary_stack * N_ori).transpose() * Lambda_primary * F_primary +
//            1 * (J_secondary_stack * N_ori * N_primary).transpose() * Lambda_secondary * F_secondary +
//            1 * (MatrixXd::Identity(robot->dof(), robot->dof()) * N_ori * N_primary * N_secondary).transpose() * robot->M() * tau;
// }


// //------------------------------------------------------------------------------

// void lambdaSmoothing(MatrixXd &_Lambda, MatrixXd &_N,
//                      shared_ptr<SaiModel::SaiModel> _robot,
//                      const MatrixXd &_projected_jacobian, const double &_e_max,
//                      const double &_e_min) {
//     // Lambda smoothing method
//     MatrixXd _Lambda_inv =
//         _projected_jacobian * _robot->MInv() * _projected_jacobian.transpose();
//     MatrixXd _Jbar, _Lambda_ns, _Lambda_s;

//     // eigendecomposition
//     // SelfAdjointEigenSolver<Matrix<double, 6, 6>>
//     // eigensolver(_Lambda_inv);
//     SelfAdjointEigenSolver<MatrixXd> eigensolver(_Lambda_inv);
//     int n_cols = 0;
//     int n_task = _projected_jacobian.rows();
//     for (int i = n_task - 1; i >= 0; --i) {
//         if (abs(eigensolver.eigenvalues()(i)) < _e_max) {
//             n_cols = i + 1;
//             break;
//         }
//     }
//     if (n_cols != 0) {
//         if (n_cols == n_task) {
//             MatrixXd U_s = eigensolver.eigenvectors();
//             MatrixXd D_s = MatrixXd::Zero(n_task, n_task);
//             VectorXd e_s = eigensolver.eigenvalues();
//             for (int i = 0; i < D_s.cols(); ++i) {
//                 if (abs(e_s(i)) < _e_min) {
//                     D_s(i, i) = 0;
//                 } else if (abs(e_s(i)) > _e_max) {
//                     D_s(i, i) = 1 / e_s(i);
//                 } else {
//                     D_s(i, i) = (1 / e_s(i)) * (0.5 + 0.5 * sin((M_PI / (_e_max - _e_min)) * (abs(e_s(i)) - _e_min) - (M_PI / 2)));
//                 }
//             }
//             _Lambda = U_s * D_s * U_s.transpose();
//             _Jbar = _robot->MInv() * _projected_jacobian.transpose() * _Lambda;
//             _N = MatrixXd::Identity(_robot->dof(), _robot->dof()) - _Jbar * _projected_jacobian;
//         } else {
//             MatrixXd U_ns = eigensolver.eigenvectors().rightCols(n_task - n_cols);
//             MatrixXd D_ns = MatrixXd::Zero(n_task - n_cols, n_task - n_cols);
//             VectorXd e_ns = eigensolver.eigenvalues().tail(n_task - n_cols);
//             for (int i = 0; i < D_ns.cols(); ++i) {
//                 D_ns(i, i) = 1 / e_ns(i);  // safe
//             }
//             MatrixXd U_s = eigensolver.eigenvectors().leftCols(n_cols);
//             MatrixXd D_s = MatrixXd::Zero(n_cols, n_cols);
//             VectorXd e_s = eigensolver.eigenvalues().head(n_cols);
//             for (int i = 0; i < D_s.cols(); ++i) {
//                 if (abs(e_s(i)) < _e_min) {
//                     D_s(i, i) = 0;
//                 } else if (abs(e_s(i)) > _e_max) {
//                     D_s(i, i) = 1 / e_s(i);
//                 } else {
//                     D_s(i, i) = (1 / e_s(i)) * (0.5 + 0.5 * sin((M_PI / (_e_max - _e_min)) * (abs(e_s(i)) - _e_min) - (M_PI / 2)));
//                 }
//                 _Lambda_ns = U_ns * D_ns * U_ns.transpose();
//                 _Lambda_s = U_s * D_s * U_s.transpose();
//                 _Lambda = _Lambda_ns + _Lambda_s;
//                 _Jbar = _robot->MInv() * _projected_jacobian.transpose() * _Lambda;
//                 _N = MatrixXd::Identity(_robot->dof(), _robot->dof()) - _Jbar * _projected_jacobian;
//             }
//         }
//     } else {
//         _Lambda = _Lambda_inv.inverse();
//         _Jbar = _robot->MInv() * _projected_jacobian.transpose() * _Lambda;
//         _N = MatrixXd::Identity(_robot->dof(), _robot->dof()) - _Jbar * _projected_jacobian;
//     }
// }

//------------------------------------------------------------------------------

// void updateSphereColor(chai3d::cShapeSphere *sphere, const int &conf_level) {
//     // std::cout << conf_level << std::endl;
//     switch (conf_level) {
//         case 2:
//             sphere->m_material->setRedFireBrick();
//             break;
//         case 1:
//             sphere->m_material->setPurpleMagenta();
//             break;
//         case 0:
//             sphere->m_material->setYellow();
//             break;
//     }
//     return;
// }