/**
 * @file verify_optitrack.cpp
 * @author William Chong (wmchong@stanford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-09-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

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

#include "SaiGraphics.h"
#include "SaiModel.h"
// #include "SaiSimulation.h"
// #include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"
#include <chrono>

bool fSimulationRunning = true;
void sighandler(int){fSimulationRunning = false;}
double DELAY = 2000; // simulation frequency in Hz
// std::shared_ptr<SaiModel::SaiModel> toro;
std::shared_ptr<SaiGraphics::SaiGraphics> graphics;

#include "redis_keys.h"

using namespace Eigen;
using namespace std;

// mutex and globals
VectorXd toro_ui_torques;

mutex mutex_torques, mutex_update;

// specify urdf and robots 
static const string toro_name = "HRP4C0";
static const string camera_name = "camera_fixed";


const std::vector<std::string> object_names = {
    "head", "lfoot", "lhand", "torso", "rfoot", "rhand"
};

// Mapping the indices from the dictionary (Python-like) to a C++ std::map
// std::map<int, int> index_mapping = {
//     {1, 5}, // head; correct
//     {2, 50}, // left foot
//     {3, 23}, // left hand
//     {4, 2},  // torso
//     {5, 47}, // right foot
//     {6, 11}  // right hand 
// };

// correct mapping 
std::map<int, int> index_mapping = {
    {1, 5}, // head; correct
    {2, 48}, // left foot; flipped
    {3, 32}, // left hand; correct
    {4, 2}, // torso; correct 
    {5, 47}, // right foot; flipped
    {6, 23}  // right hand 
};

const int N_RIGID_BODIES = 6;

void setBackgroundImage(std::shared_ptr<SaiGraphics::SaiGraphics>& graphics, const std::string& imagePath) {
    chai3d::cBackground* background = new chai3d::cBackground();
    bool fileload = background->loadFromFile(imagePath);
    if (!fileload) {
        std::cerr << "Image file loading failed: " << imagePath << std::endl;
        return;
    }
    graphics->getCamera(camera_name)->m_backLayer->addChild(background);
}

chai3d::cColorf lagrangianToColor(double lagrangian, double min_lagrangian, double max_lagrangian) {
    double normalized = (lagrangian - min_lagrangian) / (max_lagrangian - min_lagrangian);
    normalized = std::max(0.0, std::min(1.0, normalized)); // Clamp to [0, 1]

    // Blue to Red gradient
    double red = normalized;
    double blue = 1.0 - normalized;
    double green = 0.0;

    return chai3d::cColorf(red, green, blue);
}

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

int main() {
	// static const string toro_file = "./resources/model/HRP4c.urdf";
    //static const string world_file = "./resources/world/world_basic_10.urdf";
	static const string world_file = "./resources/world/world_test.urdf";
    std::cout << "Loading URDF world model file: " << world_file << endl;

    // start redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();
    redis_client.set(USER_READY_KEY, "0");

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load graphics scene
    graphics = std::make_shared<SaiGraphics::SaiGraphics>(world_file, camera_name, false);
    
    graphics->getCamera(camera_name)->setClippingPlanes(0.1, 2000);  // set the near and far clipping planes 

    // // show object link frames
    // for (auto name : object_names) {
    //     graphics->showObjectLinkFrame(true, name, 0.15);
    // }

    // setup object frames
    std::vector<Affine3d> current_object_poses;
    for (int i = 0; i < N_RIGID_BODIES; ++i) {
        current_object_poses.push_back(Affine3d::Identity());
    }

	/*------- Set up visualization -------*/
	// init redis client values

	bool conmove = true;
	bool LAGRANGIAN_BACKGROUND_MODE = true;
	bool IMAGE_BACKGROUND_MODE = false;
	// start simulation thread
	// thread sim_thread(simulation, sim);
	
	int robot_index = 0; // index to track which robot to update next
    int total_robots = 10; // total number of robots to update

    // SaiCommon::LoopTimer timer(DELAY);
    SaiCommon::LoopTimer timer(120, 1e6);  // optitrack rate 
    timer.reinitializeTimer(1e9);

	bool changed_recently = false;
	double last_background_change_time = 0.0;

	// while window is open:
	while (graphics->isWindowOpen()) {
		// toro->updateModel();
		timer.waitForNextLoop();
        const double time = timer.elapsedSimTime();

        // get object frames and update frames
        for (int i = 0; i < N_RIGID_BODIES; ++i) {
            int index = index_mapping[i + 1];
        
            Eigen::Vector3d current_position = redis_client.getEigen("0::" + std::to_string(index) + "::pos");
            Eigen::MatrixXd quaternion_matrix = redis_client.getEigen("0::" + std::to_string(index) + "::ori");

            // Create the affine transformation
            Eigen::Affine3d current_pose = Eigen::Affine3d::Identity();
            current_pose.translation() = current_position;
            current_pose.linear() = quaternionToRotationMatrix(quaternion_matrix);

            current_object_poses[i] = current_pose;

            graphics->updateObjectGraphics(object_names[i], current_pose);
        }

		// // TESTER
		// Eigen::Matrix3d test_orientation;   // 3x3 rotation matrix
		// Eigen::Vector3d test_position;      // 3D position vector

		// // Retrieve the orientation and position from Redis
		// test_orientation = redis_client.getEigen(TEST_ORI);  // Retrieves 3x3 orientation matrix
		// test_position = redis_client.getEigen(TEST_POS);     // Retrieves 3x1 position vector

		// // Create an Affine3d object for the pose
		// Eigen::Affine3d test_pose;
		// test_pose.linear() = test_orientation;  // Set the rotation part
		// test_pose.translation() = test_position;  // Set the translation part
		// //Quaterniond quat(test_orientation);

		// // Now update the graphics for the object named "tester"
		// graphics->updateObjectGraphics("tester", test_pose);

		// Optionally, show the frame for the object to visualize its orientation
		// graphics->showObjectLinkFrame(true, "tester", 0.05);

		// Get the Lagrangian value from Redis

		// if (LAGRANGIAN_BACKGROUND_MODE) {
		// double lagrangian = stod(redis_client.get(LAGRANGIAN));
		// chai3d::cColorf backgroundColor = lagrangianToColor(lagrangian, -50.0, 200.0);
		// double red = backgroundColor.getR();
    	// double green = backgroundColor.getG();
   		// double blue = backgroundColor.getB();
		// graphics->setBackgroundColor(red, green, blue);
		// }
        
        // // Update primary robot graphics
        // //graphics->updateRobotGraphics(toro_name, redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
        
        // // Update one robot graphics per iteration
        // // std::string name = "HRP4C" + std::to_string(robot_index);
        // // graphics->updateRobotGraphics(name, redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
        // int index = (index + 1) % total_robots;
		// for (int robot_index = 0; robot_index < 10; ++robot_index) {
		// 	graphics->updateRobotGraphics("HRP4C" + std::to_string(robot_index), redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
		// }

		// if (index == 8) {
		// 	changed_recently = false; // better logic for this
		// }
		
		// Eigen::Vector3d ra_end_effector_pos = redis_client.getEigen("1::11::pos");
		// Eigen::Vector3d la_end_effector_pos = redis_client.getEigen("1::23::pos");

		// Calculate the L2 norm of the difference between the end effectors
		// double l2_norm = (ra_end_effector_pos - la_end_effector_pos).norm();

		double last_background_change_time = 0.0; // Initialize the last background change time

		// Eigen::Vector3d ground_threshold = redis_client.getEigen(GROUND);
    	// double ground_z = ground_threshold.z();  // Extract the z component

		// Eigen::Vector3d current_rleg_pos = redis_client.getEigen("1::47::pos");
		// Eigen::Vector3d current_lleg_pos = redis_client.getEigen("1::50::pos");
		// //std::cout << "right leg:" << current_rleg_pos.z() << std::endl;

		// // // Check if both legs are off the ground relative to their initial positions
		// bool isJumping = (current_rleg_pos.z() - ground_z > -0.08) && (current_lleg_pos.z() - ground_z > -0.08);
		// double rleg_ground_difference = current_rleg_pos.z() - ground_z;
		// std::cout << "foot " << rleg_ground_difference << std::endl;


		// if (isJumping) {
		// 	graphics->setBackgroundColor(1.0, 1.0, 1.0);  // White RGB
		// }

		// if (l2_norm < CLAP_THRESHOLD) {
		// 	if (!changed_recently) {
		// 		int current_clap_count = redis_client.getInt(CLAP_KEY);
		// 		redis_client.setInt(CLAP_KEY, current_clap_count + 1);

		// 		int background_index = (current_clap_count + 1) % background_paths.size();
		// 		if (IMAGE_BACKGROUND_MODE) {
		// 			setBackgroundImage(graphics, background_paths[background_index]);
		// 		}
		// 		// Update the last background change time
		// 		last_background_change_time = time;
		// 		changed_recently = true;
		// 	}
		// }

		// if (l2_norm < CLAP_THRESHOLD) {
			//graphics->setBackgroundColor(1.0, 1.0, 1.0);  // White RGB
		// }

		// Retrieve the head position
		// Eigen::Vector3d head_pos = redis_client.getEigen("Sai::sim::toro::head_pos");

		// Check if both hands are above the head
		// bool hands_above_head = (head_pos.z() - ra_end_effector_pos.z() < HANDS_ABOVE_HEAD_THRESHOLD ||
								// head_pos.z() - la_end_effector_pos.z() < HANDS_ABOVE_HEAD_THRESHOLD);
		// redis_client.setBool(HANDS_ABOVE_HEAD_KEY, hands_above_head);

		// if (hands_above_head) {
			//graphics->setBackgroundColor(0.0, 0.0, 0.0);  // Black RGB
		// }

		// double hands_above = head_pos.z() - ra_end_effector_pos.z();
		//std::cout << hands_above << std::endl;
		
		//std::cout << head_pos.z() - ra_end_effector_pos.z() << std::endl;
		// Set transparency for odd-numbered robots if hands are above the head
		//if (hands_above < 0.1) {
		// if (true) {
		// 	graphics->showTransparency(true, "toro0", 0.0);
		// 	graphics->showTransparency(true, "toro1", 0.0);
		// 	graphics->showTransparency(true, "toro2", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro3", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro4", 0.0);
		// 	graphics->showTransparency(true, "toro5", 1.0);
		// 	graphics->showTransparency(true, "toro6", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro7", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro8", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro9", 0.0); // Fully transparent
		// } 
		// else if (0.3 > hands_above > 0.1) {
		// 	graphics->showTransparency(true, "toro0", 0.0);
		// 	graphics->showTransparency(true, "toro1", 0.0);
		// 	graphics->showTransparency(true, "toro2", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro3", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro4", 1.0);
		// 	graphics->showTransparency(true, "toro4", 1.0);
		// 	graphics->showTransparency(true, "toro5", 1.0);
		// 	graphics->showTransparency(true, "toro6", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro7", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro8", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro9", 0.0); // Fully transparent
		// } 
		// else if (0.6 > hands_above > 0.3) {
		// 	graphics->showTransparency(true, "toro0", 1.0);
		// 	graphics->showTransparency(true, "toro1", 0.0);
		// 	graphics->showTransparency(true, "toro2", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro3", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro4", 1.0);
		// 	graphics->showTransparency(true, "toro5", 0.0);
		// 	graphics->showTransparency(true, "toro6", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro7", 0.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro8", 1.0); // Fully transparent
		// 	graphics->showTransparency(true, "toro9", 0.0); 
		// }
		// else {
		// 	graphics->showTransparency(true, "toro0", 1.0);
		// 	graphics->showTransparency(true, "toro1", 1.0);
		// 	graphics->showTransparency(true, "toro2", 1.0); 
		// 	graphics->showTransparency(true, "toro3", 1.0); 
		// 	graphics->showTransparency(true, "toro4", 1.0);
		// 	graphics->showTransparency(true, "toro5", 1.0);
		// 	graphics->showTransparency(true, "toro6", 1.0); 
		// 	graphics->showTransparency(true, "toro7", 1.0);
		// 	graphics->showTransparency(true, "toro8", 1.0); 
		// 	graphics->showTransparency(true, "toro9", 1.0); 
		// }
    
        // {
		// 	lock_guard<mutex> lock(mutex_update);
			
		// }

		// //if (redis_client.getBool(GLOBAL_CAM) && conmove) {
		// if (false) {
		// 	Eigen::Vector3d head_pos;
		// 	head_pos << 2.0, -0.8, 6.0;
		// 	Eigen::Vector3d head_vert_axis;
		// 	head_vert_axis << 0.0, 0.0, 1.0;
		// 	Eigen::Vector3d head_look_at;
		// 	head_look_at << 0.0, 0.0, 0.0;

		// 	// Create the rotation matrix
		// 	Eigen::Matrix3d rotation;
		// 	rotation.col(2) = head_vert_axis.normalized(); // z-axis
		// 	rotation.col(0) = head_look_at.normalized(); // x-axis
		// 	rotation.col(1) = rotation.col(2).cross(rotation.col(0)).normalized(); // y-axis

		// 	// Create the Affine3d object
		// 	Eigen::Affine3d camera_pose = Eigen::Affine3d::Identity();
		// 	camera_pose.translation() = head_pos;
		// 	camera_pose.linear() = rotation;

		// 	// Call setCameraPose with the correct arguments
		// 	graphics->setCameraPose(camera_name, camera_pose);
		// 	conmove = false;
		// } 

		// else {
		// 	// Retrieve the head position, vertical axis, and look-at direction
		// 	Eigen::Vector3d head_pos = redis_client.getEigen(HEAD_POS);
		// 	Eigen::Vector3d head_vert_axis = redis_client.getEigen(HEAD_VERT_AXIS);
		// 	Eigen::Vector3d head_look_at = redis_client.getEigen(HEAD_LOOK_AT);

		// 	// Create the rotation matrix
		// 	Eigen::Matrix3d rotation;
		// 	rotation.col(2) = head_vert_axis.normalized(); // z-axis
		// 	rotation.col(0) = head_look_at.normalized(); // x-axis
		// 	rotation.col(1) = rotation.col(2).cross(rotation.col(0)).normalized(); // y-axis

		// 	// Create the Affine3d object
		// 	Eigen::Affine3d camera_pose = Eigen::Affine3d::Identity();
		// 	camera_pose.translation() = head_pos;
		// 	camera_pose.linear() = rotation;

		// 	// Call setCameraPose with the correct arguments
		// 	graphics->setCameraPose(camera_name, camera_pose);
		// 	conmove = true;
		// }

		
		graphics->renderGraphicsWorld();
		// {
		// 	lock_guard<mutex> lock(mutex_torques);
			
		// 	toro_ui_torques = graphics->getUITorques(toro_name);
		// }
	}

    // stop simulation
	fSimulationRunning = false;
	// sim_thread.join();

	return 0;
}

// //------------------------------------------------------------------------------
// void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim) {
//     // fSimulationRunning = true;

//     // create redis client
//     auto redis_client = SaiCommon::RedisClient();
//     redis_client.connect();

//     // create a timer
//     double sim_freq = 2000;
//     SaiCommon::LoopTimer timer(sim_freq);

//     sim->setTimestep(1.0 / sim_freq);
//     sim->enableGravityCompensation(true);

//     sim->enableJointLimits(toro_name);

//     while (fSimulationRunning) {
//         timer.waitForNextLoop();
//         const double time = timer.elapsedSimTime();

//         VectorXd toro_control_torques = redis_client.getEigen(TORO_JOINT_TORQUES_COMMANDED_KEY);
//         {
//             lock_guard<mutex> lock(mutex_torques);
//             sim->setJointTorques(toro_name, toro_control_torques + toro_ui_torques);
//         }
//         sim->integrate();

// 		VectorXd robot_q = sim->getJointPositions(toro_name);
//     	VectorXd robot_dq = sim->getJointVelocities(toro_name);

//         //robot_dq = 0.1 * VectorXd::Ones(robot_dq.size()) * sin(time);

//         // Get the mass matrix
// 		toro->updateModel();
//         MatrixXd robot_M = toro->M();
//         VectorXd g = toro->jointGravityVector();
//         double kinetic_energy = 0.5 * robot_dq.transpose() * robot_M * robot_dq;
// 		// Eigen::Vector3d comPosition = toro->comPosition();
// 		// Eigen::Affine3d toro_base_transform_in_world = toro->transformInWorld("RL_KOSY_L1");
// 		// Eigen::Vector3d comPositionInWorld = toro_base_transform_in_world.linear() * comPosition + toro_base_transform_in_world.translation();
// 		// Eigen::Vector3d ground = redis_client.getEigen("1::2::pos");
// 		// double potential_energy = 90 * (robot_q(2)+ground_threshold.z()) * 9.8;
// 		Vector3d ground = Vector3d::Zero();
// 		double potential_energy = 90 * (ground[1]) * 9.8;
		
// 		std::cout<<"see me: "<< robot_q.head(3).transpose()<<std::endl;
//         double lagrangian = kinetic_energy - potential_energy;
        
// 		redis_client.set(KINETIC, std::to_string(kinetic_energy));
// 		redis_client.set(POTENTIAL, std::to_string(potential_energy));
// 		redis_client.set(LAGRANGIAN, std::to_string(lagrangian));

//         redis_client.setEigen(TORO_JOINT_ANGLES_KEY, robot_q);
//         redis_client.setEigen(TORO_JOINT_VELOCITIES_KEY, robot_dq);

//         {
//             lock_guard<mutex> lock(mutex_update);
//         }
//     }
//     timer.stop();
//     cout << "\nSimulation loop timer stats:\n";
//     timer.printInfoPostRun();
// }

