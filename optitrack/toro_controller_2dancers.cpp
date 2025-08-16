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

bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;
using namespace Optitrack;

// specify urdf and robots
const string robot_file = "./resources/model/HRP4c.urdf";
const std::vector<string> robot_name = {"HRP4C0", "HRP4C1"};
const string camera_name = "camera_fixed";
const std::string yaml_fname = "./resources/controller_settings_2_dancers.yaml";

// globals
std::vector<VectorXd> nominal_posture;
// std::vector<VectorXd control_torques;
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

const double MOTION_SF = 1.0;
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

void control(std::shared_ptr<Human> human,
             std::shared_ptr<SaiModel::SaiModel> robot,
             OptitrackData& optitrack_data,
             SimBodyData& sim_body_datas,
             ControllerData& controller_data,
             const VectorXd& q_init,
             const int robot_id);

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

    // initialize redis 
    redis_client.setInt(USER_READY_KEY, 0);

    // setup struct 
    std::vector<OptitrackData> optitrack_data_vectorized;
    std::vector<SimBodyData> sim_body_data_vectorized;
    std::vector<ControllerData> controller_data_vectorized;

    auto optitrack_data = OptitrackData();
    // auto sim_body_data = SimBodyData();
    auto controller_data = ControllerData();

    // parse yaml controller settings 
    YAML::Node config = YAML::LoadFile(yaml_fname);

    // optitrack settings 
    YAML::Node current_node = config["optitrack"];
    std::vector<std::string> body_part_names = current_node["body_part_names"].as<std::vector<std::string>>();
    std::vector<int> optitrack_ids = current_node["indices"].as<std::vector<int>>();
    std::vector<int> human_ids = current_node["human_ids"].as<std::vector<int>>();

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

    auto second_optitrack_data = optitrack_data;

    optitrack_data_vectorized.push_back(optitrack_data);
    optitrack_data_vectorized.push_back(second_optitrack_data);

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
    // redis_client.setDouble(Z_VIEWING_ANGLE, 180);
    // R_optitrack_to_sai *= AngleAxisd(90 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix();

    std::vector<std::shared_ptr<SaiModel::SaiModel>> robot;
    for (int i = 0; i < 2; ++i) {
        robot.push_back(std::make_shared<SaiModel::SaiModel>(robot_file, false));
        robot[i]->updateModel();
    }    

    // create human class for tracking 
    std::vector<std::shared_ptr<Optitrack::Human>> human;
    for (int i = 0; i < 2; ++i) {
        human.push_back(std::make_shared<Optitrack::Human>(controller_data.control_links));
        human[i]->setMultiRotationReference(controller_data.control_links, {controller_data.control_links.size(), R_optitrack_to_sai});
    }

    auto second_controller_data = controller_data;
    controller_data_vectorized.push_back(controller_data);
    controller_data_vectorized.push_back(second_controller_data);

    sim_body_data_vectorized.push_back(SimBodyData());
    sim_body_data_vectorized.push_back(SimBodyData());

    // nominal postures 
    // VectorXd q_init(robot[0]->dof());
    // q_init << 0, 0, 0, 0, 0, 0, 
    //             0, -0.1, -0.25, 0.5, -0.25, 0.1, 
    //             0, 0.1, -0.25, 0.5, -0.25, -0.1, 
    //             0, 0,
    //             -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
    //             -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
    //             0, 0;

    // nominal_posture.push_back(q_init);

    // q_init << 0, 0.5, 0, 0, 0, 0, 
    //             0, -0.1, -0.25, 0.5, -0.25, 0.1, 
    //             0, 0.1, -0.25, 0.5, -0.25, -0.1, 
    //             0, 0,
    //             -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
    //             -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
    //             0, 0;
    // nominal_posture.push_back(q_init);

    for (int i = 0; i < 2; ++i) {
        nominal_posture.push_back(redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[i]));
    }

    // start controller threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::thread(control, human[i], robot[i], std::ref(optitrack_data_vectorized[i]), std::ref(sim_body_data_vectorized[i]), std::ref(controller_data_vectorized[i]), nominal_posture[i], i));
    }
    // thread second_human_control_thread(control, human[1], robot[1], std::ref(optitrack_data_vectorized[1]), std::ref(sim_body_data_vectorized[1]), std::ref(controller_data_vectorized[1]), nominal_posture[1], 1);

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();  // Blocks until the thread finishes execution
    }

	return 0;
}

//------------------------------------------------------------------------------
void control(std::shared_ptr<Optitrack::Human> human,
             std::shared_ptr<SaiModel::SaiModel> robot,
             OptitrackData& optitrack_data,
             SimBodyData& sim_body_datas,
             ControllerData& controller_data,
             const VectorXd& q_init,
             const int robot_id) {

    // create redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    std::cout << "[Robot " << robot_id << "] Reading joint angles from key: " << MULTI_TORO_JOINT_ANGLES_KEY[robot_id] << std::endl;
    auto q_read = redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[robot_id]);
    std::cout << "Joint angles size: " << q_read.size() << std::endl;

    std::cout << "[Robot " << robot_id << "] Reading velocities from key: " << MULTI_TORO_JOINT_VELOCITIES_KEY[robot_id] << std::endl;
    auto dq_read = redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[robot_id]);
    std::cout << "Joint velocities size: " << dq_read.size() << std::endl;


    // reset user ready key 
    redis_client.setInt(MULTI_USER_READY_KEY[robot_id], 0);
    redis_client.setInt(MULTI_RESET_ROBOT_KEY[robot_id], 0);

	// update robot model and initialize control vectors
    robot->setQ(redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[robot_id]));
    robot->setDq(redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[robot_id]));
    std::cout << "[Robot " << robot_id << "] robot->q().head(6): " << robot->q().head(6).transpose() << std::endl;
    robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

    auto sim_body_data = SimBodyData();

    // create task mappings 
    std::map<std::string, std::shared_ptr<SaiPrimitives::MotionForceTask>> tasks;

    for (int i = 0; i < controller_data.control_links.size(); ++i) {
        Affine3d compliant_frame = Affine3d::Identity();
        compliant_frame.translation() = controller_data.control_points[i];
        tasks[controller_data.control_links[i]] = std::make_shared<SaiPrimitives::MotionForceTask>(robot,
                                                                                                    controller_data.control_links[i],
                                                                                                    compliant_frame,
                                                                                                    controller_data.control_links[i]);
        
        tasks[controller_data.control_links[i]]->disableInternalOtg();  // TESTING 
        tasks[controller_data.control_links[i]]->setDynamicDecouplingType(SaiPrimitives::FULL_DYNAMIC_DECOUPLING);
        // tasks[controller_data.control_links[i]]->handleAllSingularitiesAsType1(true);
        // tasks[controller_data.control_links[i]]->disableSingularityHandling();
        if (controller_data.control_links[i] == "trunk_rz" 
                || controller_data.control_links[i] == "neck_link2"
                || controller_data.control_links[i] == "ra_link4" 
                || controller_data.control_links[i] == "la_link4") {
            tasks[controller_data.control_links[i]]->disableSingularityHandling();
        }
        if (controller_data.control_links[i] == "ra_end_effector" || controller_data.control_links[i] == "la_end_effector") {
            tasks[controller_data.control_links[i]]->setSingularityHandlingBounds(1e-3, 1e-2);  // might need to change based on the control link (to test)
        }
        // tasks[controller_data.control_links[i]]->setSingularityHandlingBounds(1e-3, 1e-2);
        tasks[controller_data.control_links[i]]->setPosControlGains(400, 40, 0);
        tasks[controller_data.control_links[i]]->setOriControlGains(400, 40, 0);
    }

    auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
    joint_task->disableInternalOtg();  // TESTING 
    // VectorXd q_desired = robot->q();
    // q_desired << 0, 0, 0, 0, 0, 0, 
    //             0, -0.1, -0.25, 0.5, -0.25, 0.1, 
    //             0, 0.1, -0.25, 0.5, -0.25, -0.1, 
    //             0, 0,
    //             -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
    //             -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
    //             0, 0;
	joint_task->setGains(400, 40, 0);
    joint_task->setDynamicDecouplingType(SaiPrimitives::DynamicDecouplingType::FULL_DYNAMIC_DECOUPLING);
	joint_task->setGoalPosition(q_init);  
    // nominal_posture.push_back(q_desired);

    // create robot controller
    std::vector<shared_ptr<SaiPrimitives::TemplateTask>> task_list;
    for (auto task_name : controller_data.control_links) {
        task_list.push_back(tasks[task_name]);
    }
    task_list.push_back(joint_task);
	auto robot_controller = std::make_unique<SaiPrimitives::RobotController>(robot, task_list);

    // initialize
    int state = INIT;

    bool first_loop = true;
    const int n_calibration_samples = 1 * 1000;  // N second of samples
    int n_samples = 0;
    VectorXd robot_control_torques = VectorXd::Zero(dof);
    VectorXd prev_control_torques = VectorXd::Zero(dof);

	// create a loop timer
    runloop = true;
	double control_freq = 500;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (runloop) {
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();

        // update robot model
        robot->setQ(redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[robot_id]));
        robot->setDq(redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[robot_id]));
        robot->updateModel();

        // read the reset state 
        int reset_robot = redis_client.getInt(MULTI_RESET_ROBOT_KEY[robot_id]);
        if (reset_robot) {
            state = RESET;
            redis_client.setInt(MULTI_RESET_ROBOT_KEY[robot_id], 0);
        }

        // read optitrack input and store in optitrack struct 
        if (state == CALIBRATION || state == TRACKING) {

            for (auto it = optitrack_data.body_index_mapping.begin(); it != optitrack_data.body_index_mapping.end(); ++it) {

                std::string body_part_name = it->first;
                int index = it->second;
                Vector3d current_position = redis_client.getEigen(std::to_string(optitrack_data.human_ids[robot_id]) + "::" + std::to_string(index) + "::pos");
                MatrixXd quaternion_matrix = redis_client.getEigen(std::to_string(optitrack_data.human_ids[robot_id]) + "::" + std::to_string(index) + "::ori");
                std::cout << "Position: " << current_position.transpose() << std::endl;
                std::cout << "Orientation size: " << quaternion_matrix.size() << std::endl;

                if (quaternion_matrix.size() != 4) {
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

                // Overwrite map for pose 
                optitrack_data.current_pose[body_part_name] = current_pose;

            }
            // // If needed, store in other vectors as well
            //     if (i <= 3) {
            //         current_primary_poses.push_back(current_pose);
            //     } else {
            //         current_secondary_poses.push_back(current_pose);
            //     }
        }

        // // print out
        // for (int i = 1; i < NUM_RIGID_BODIES; ++i) {
        //     std::cout << "Rigid body " << i << " position: " << current_link_poses[i].translation().transpose() << std::endl;
        //     std::cout << "Rigid body " << i << " orientation: " << current_link_poses[i].rotation() << std::endl;
        // }
        // perform robustness checks with the optitrack input 

        if (state == INIT) {
            // start robot at default configuration and hold the posture
            joint_task->setGoalPosition(q_init);
            N_prec.setIdentity();
            joint_task->updateTaskModel(N_prec);
            robot_control_torques = joint_task->computeTorques();

            if (joint_task->goalPositionReached(1e-2)) {
                if (redis_client.getInt(MULTI_USER_READY_KEY[robot_id]) == 1) {
                    // state = CALIBRATION;
                    state = TEST;
                    first_loop = true;
                    n_samples = 0;
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
            bool user_ready = redis_client.getBool(MULTI_USER_READY_KEY[robot_id]);

            if (user_ready) {
                // recalibrate 
                std::vector<std::string> link_names;
                std::vector<Affine3d> link_poses;
                for (auto it = optitrack_data.current_pose.begin(); it != optitrack_data.current_pose.end(); ++it) {
                    link_names.push_back(it->first);
                    link_poses.push_back(it->second);
                }
                human->calibratePose(link_names, link_poses, n_calibration_samples, first_loop);
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
                    // Eigen::Vector3d initial_rleg_pos = human->getInitialPose("RL_foot").translation();
                    // Eigen::Vector3d initial_lleg_pos = human->getInitialPose("LL_foot").translation();

                    // // Calculate the ground level as the average of the z-coordinates of the initial leg positions
                    // double ground_z = (initial_rleg_pos.z() + initial_lleg_pos.z()) / 2.0;
                    // Eigen::Vector3d ground_pos(0, 0, ground_z);

                    // // Set the ground key in Redis
                    // redis_client.setEigen(GROUND, ground_pos);

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
            // want to measure relative motion in optitrack frame
            robot_controller->updateControllerTaskModels();

            // manually set type 1 posture for all tasks to be the starting posture 
            for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                it->second->setType1Posture(q_init);
            }

            // update simulation current pose 
            std::vector<Affine3d> sim_current_pose;
            for (int i = 0; i < controller_data.control_links.size(); ++i) {
                Affine3d current_pose = Affine3d::Identity();
                current_pose.translation() = robot->positionInWorld(controller_data.control_links[i], controller_data.control_points[i]);
                current_pose.linear() = robot->rotationInWorld(controller_data.control_links[i]);
                sim_current_pose.push_back(current_pose);
            }

            // package optitrack poses for relative pose
            std::vector<Affine3d> optitrack_current_pose;
            for (int i = 0; i < controller_data.control_links.size(); ++i) {
                optitrack_current_pose.push_back(optitrack_data.current_pose[controller_data.control_links[i]]);
            }

             // -------- set task goals and compute control torques
            robot_control_torques.setZero();

            auto relative_poses = human->relativePose(controller_data.control_links, optitrack_current_pose);  // returns back relative poses in order of control links 

            int i = 0;
            for (auto name : controller_data.control_links) {

                Vector3d desired_position = sim_body_data.starting_pose[name].translation() + MOTION_SF * relative_poses[i].translation();
                Matrix3d desired_orientation = relative_poses[i].linear() * sim_body_data.starting_pose[name].linear();

                tasks[name]->setGoalPosition(desired_position);
                tasks[name]->setGoalOrientation(desired_orientation);
                i++;
            }

            robot_control_torques = robot_controller->computeControlTorques() + robot->coriolisForce();

            // end 

        } else if (state == TEST) {
            // std::cout << "Test\n";
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
            tasks["hip_base"]->setGoalPosition(sim_body_data.starting_pose["hip_base"].translation() + Vector3d(0, 0, 0.1));
            // tasks["hip_base"]->setGoalPosition(sim_body_data.starting_pose["hip_base"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));
            // tasks["RL_foot"]->setGoalPosition(sim_body_data.starting_pose["RL_foot"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));
            // tasks["LL_foot"]->setGoalPosition(sim_body_data.starting_pose["LL_foot"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));

            // tasks["ra_end_effector"]->setGoalPosition(sim_body_data.starting_pose["ra_end_effector"].translation() + 0.1 * Vector3d(sin(2 * M_PI * time), sin(2 * M_PI * time), sin(2 * M_PI * time)));
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

		// //------ compute the final torques
		// {
		// 	lock_guard<mutex> lock(mutex_torques);
		// 	control_torques = robot_control_torques;
		// }

        if (isnan(robot_control_torques(0))) {
            // throw runtime_error("nan torques");
            std::cout << "nan torques: setting to previous torque\n";
            robot_control_torques = prev_control_torques;
        }

        prev_control_torques = robot_control_torques;
        
        // debug test 
        // robot_control_torques.setZero();
        // robot_control_torques(0) = 0.05;

         // std::cout << robot->q().head(6).transpose() << "\n";

        redis_client.setEigen(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[robot_id], robot_control_torques);

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
    for (int i = 0; i < 2; ++i) {
        redis_client.setEigen(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[i], 0 * redis_client.getEigen(MULTI_TORO_JOINT_TORQUES_COMMANDED_KEY[i]));
    }
	
}
