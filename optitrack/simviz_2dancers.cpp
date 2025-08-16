/**
 * @file simviz.cpp
 * @brief Simulation and visualization of dancing toro 
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
#include "SaiSimulation.h"
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"
#include <chrono>
#include <yaml-cpp/yaml.h>

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

bool fSimulationRunning = true;
void sighandler(int){fSimulationRunning = false;}
double DELAY = 2000; // simulation frequency in Hz
std::shared_ptr<SaiModel::SaiModel> hannah;
std::shared_ptr<SaiModel::SaiModel> tracy;
std::shared_ptr<SaiGraphics::SaiGraphics> graphics;

#include "redis_keys.h"

using namespace Eigen;
using namespace std;

// mutex and globals
VectorXd hannah_ui_torques;
VectorXd tracy_ui_torques;

mutex mutex_torques, mutex_update;

// specify urdf and robots 
static const string hannah_name = "HRP4C0";
static const string tracy_name = "HRP4C1";
static const string camera_name = "camera_fixed";
const std::string yaml_fname = "./resources/controller_settings_multi_dancers.yaml";

static const string toro_file = "./resources/model/HRP4c.urdf";
static const string human_file = "./resources/model/human.urdf";
//static const string world_file = "./resources/world/world_basic_10.urdf";
static const string world_file = "./resources/world/world_basic_2.urdf";

bool DEBUG = false;
std::vector<int> limited_joints;
VectorXd hannah_q_desired(38);
VectorXd tracy_q_desired(38);

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

// simulation thread
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim,
				const std::vector<double>& lower_limit,
				const std::vector<double>& upper_limit);

void computeEnergy();

int main() {
    std::cout << "Loading URDF world model file: " << world_file << endl;

	// parse yaml controller settings 
    YAML::Node config = YAML::LoadFile(yaml_fname);

    // optitrack settings 
    YAML::Node current_node = config["optitrack"];
    std::vector<std::string> body_part_names = current_node["body_part_names"].as<std::vector<std::string>>();
	DEBUG = current_node["debug"].as<bool>();
	limited_joints = current_node["limited_joints"].as<std::vector<int>>();

	// print settings
	std::cout << "Debug mode: " << DEBUG << "\n";
	std::cout << "Limited joints: ";
	for (auto joint : limited_joints) {
		std::cout << joint << ", ";
	}

    // start redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load graphics scene
    graphics = std::make_shared<SaiGraphics::SaiGraphics>(world_file, "Humanoid Motion Mapping", false);
    graphics->getCamera(camera_name)->setClippingPlanes(0.1, 2000);  // set the near and far clipping planes 
	// graphics->setMirrorHorizontal(camera_name, true);

	int total_robots = 2; // total number of robots to update (BEFORE WAS 10)

    // load robots
    hannah = std::make_shared<SaiModel::SaiModel>(toro_file, false);
    hannah->updateModel();

	tracy = std::make_shared<SaiModel::SaiModel>(toro_file, false);
	tracy->updateModel();

    hannah_ui_torques = VectorXd::Zero(hannah->dof());
	tracy_ui_torques = VectorXd::Zero(tracy->dof());
	// SET the initial q -- define manual here to stand

	// hannah_q_desired << 0, 0.75, 0, 0, 0, 0, 0, -0.1, -0.25, 0.5, -0.25, 0.1, 0, 0.1, -0.25, 0.5, -0.25, -0.1, 0, 0, -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 0, 0;
    // tracy_q_desired << 0, -0.75, 0, 0, 0, 0, 0, -0.1, -0.25, 0.5, -0.25, 0.1, 0, 0.1, -0.25, 0.5, -0.25, -0.1, 0, 0, -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 0, 0;

	hannah_q_desired << 0, 0.75, 0, 0, 0, 0, 
					0, -0.1, -0.25, 0.5, -0.25, 0, 0.1, 
					0, 0.1, -0.25, 0.5, -0.25, 0, -0.1, 
					0, 0,
					-0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
					-0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
					0, 0;

	tracy_q_desired << 0, -0.75, 0, 0, 0, 0, 
					0, -0.1, -0.25, 0.5, -0.25, 0, 0.1, 
					0, 0.1, -0.25, 0.5, -0.25, 0, -0.1, 
					0, 0,
					-0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
					-0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
					0, 0;

	// hannah_q_desired(24) = -1.0;
	// hannah_q_desired(25) = 0.6;
	// hannah_q_desired(26) = -0.1;
	// hannah_q_desired(25) = 0;
	// hannah_q_desired(31) = 0;
	// hannah_q_desired(32) = 0;

	// tracy_q_desired(24) = -1.0;
	// tracy_q_desired(25) = 0.6;
	// tracy_q_desired(26) = -0.1;
	// tracy_q_desired(25) = 0;
	// tracy_q_desired(31) = 0;
	// tracy_q_desired(32) = 0;

	hannah->setQ(hannah_q_desired);
	hannah->updateModel();

	tracy->setQ(tracy_q_desired);	
	tracy->updateModel();
    
	// hannah_joint_task->setGoalPosition(hannah_q_desired);
	// tracy_joint_task->setGoalPosition(tracy_q_desired);
	// graphics->addUIForceInteraction(toro_name);

	// load simulation world
	auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
	
	sim->setJointPositions(hannah_name, hannah->q());
	sim->setJointVelocities(hannah_name, hannah->dq());

	sim->setJointPositions(tracy_name, tracy->q());
	sim->setJointVelocities(tracy_name, tracy->dq());
	
	// for (int robot_index = 0; robot_index < 10; ++robot_index) {
	// 	if (robot_index % 2 == 0) {  // Even index robots (Hannah)
	// 		sim->setJointPositions("HRP4C" + std::to_string(robot_index), hannah_q_desired);
	// 	} else {  // Odd index robots (Tracy)
	// 		sim->setJointPositions("HRP4C" + std::to_string(robot_index), tracy_q_desired);
	// 	}
	// }

	// for (int robot_index = 0; robot_index < total_robots; ++robot_index) {
	// 	if (robot_index % 2 == 0) {  // Even index robots (Hannah)
	// 		sim->setJointPositions("HRP4C" + std::to_string(robot_index), hannah_q_desired);
	// 	} else {  // Odd index robots (Tracy)
	// 		sim->setJointPositions("HRP4C" + std::to_string(robot_index), tracy_q_desired);
	// 	}
	// }

    // set co-efficient of restition to zero for force control
    sim->setCollisionRestitution(0.0);

    // set co-efficient of friction
    sim->setCoeffFrictionStatic(0.0);
    sim->setCoeffFrictionDynamic(0.0);

	// parse joint limits 
	std::vector<double> lower_joint_limits, upper_joint_limits;
	auto joint_limits = hannah->jointLimits();
	for (auto limit : joint_limits) {
		lower_joint_limits.push_back(limit.position_lower);
		upper_joint_limits.push_back(limit.position_upper);
	}

	/*------- Set up visualization -------*/
	// init redis client values 
	//CHANGED
	// redis_client.setEigen(HANNAH_TORO_JOINT_ANGLES_KEY, hannah->q()); 
	// redis_client.setEigen(HANNAH_TORO_JOINT_VELOCITIES_KEY, hannah->dq()); 
	// redis_client.setEigen(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY, 0 * hannah->q());

	redis_client.setEigen(TRACY_TORO_JOINT_ANGLES_KEY, tracy->q()); 
	redis_client.setEigen(TRACY_TORO_JOINT_VELOCITIES_KEY, tracy->dq()); 
	redis_client.setEigen(TRACY_TORO_JOINT_TORQUES_COMMANDED_KEY, 0 * tracy->q());

	string link_name = "neck_link2"; // head link
	Eigen::Affine3d transform = hannah->transformInWorld(link_name); // p_base = T * p_link
	MatrixXd rot = transform.rotation();
	VectorXd pos = transform.translation();
	VectorXd vert_axis = rot.col(2); // straight up from head (pos z)
	VectorXd lookat = rot.col(0); // straight ahead of head (pos x)

	VectorXd offset(3);
	offset << -2.8, 0.0, -1.1; // x = 1.6
	pos += offset;

	redis_client.setEigen(HEAD_POS, pos);
	redis_client.setEigen(HEAD_VERT_AXIS, vert_axis);
	redis_client.setEigen(HEAD_LOOK_AT, lookat + pos);

	bool conmove = true;
	bool LAGRANGIAN_BACKGROUND_MODE = false;
	bool IMAGE_BACKGROUND_MODE = false;
	
	redis_client.set(LAGRANGIAN, std::to_string(0));
	redis_client.set(RESET_SIM_KEY, "0");

	// start simulation thread
	thread sim_thread(simulation, sim, lower_joint_limits, upper_joint_limits);
	thread compute_thread(computeEnergy);
	
	int robot_index = 0; // index to track which robot to update next

    SaiCommon::LoopTimer timer(120, 1e6);
    timer.reinitializeTimer(1e9);

	bool changed_recently = false;
	double last_background_change_time = 0.0; 

	// while window is open:
	while (graphics->isWindowOpen() && fSimulationRunning) {
		timer.waitForNextLoop();
        const double time = timer.elapsedSimTime();

		// Get the Lagrangian value from Redis

		if (true) {
			// double lagrangian = 1;
			// if (stod(redis_client.get(HANNAH_HUMAN_KINETIC_KEY)) != 0) {
				// lagrangian = stod(redis_client.get(HANNAH_ROBOT_KINETIC_KEY)) / stod(redis_client.get(HANNAH_HUMAN_KINETIC_KEY));
			// } 
			double lagrangian = stod(redis_client.get(HANNAH_ROBOT_KINETIC_KEY));
			// std::cout << lagrangian << "\n";
			chai3d::cColorf backgroundColor = lagrangianToColor(lagrangian, 0.0, 250);
			double red = backgroundColor.getR();
			double green = backgroundColor.getG();
			double blue = backgroundColor.getB();
			// graphics->setBackgroundColor(red, green, blue);
			graphics->setBackgroundColor(0, 0, 0);
			redis_client.set("red", std::to_string(red));
			redis_client.set("green", std::to_string(green));
			redis_client.set("blue", std::to_string(blue));
		}
        
        // Update primary robot graphics
        //graphics->updateRobotGraphics(toro_name, redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
        
        // Update one robot graphics per iteration
        // std::string name = "HRP4C" + std::to_string(robot_index);
        // graphics->updateRobotGraphics(name, redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
        int index = (index + 1) % total_robots;
		// for (int robot_index = 0; robot_index < 10; ++robot_index) {
		// 	graphics->updateRobotGraphics("HRP4C" + std::to_string(robot_index), redis_client.getEigen(TORO_JOINT_ANGLES_KEY));
		// }
		// for (int robot_index = 0; robot_index < total_robots; ++robot_index) {
		// 	graphics->updateRobotGraphics("HRP4C" + std::to_string(robot_index), redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[robot_index]));
		// }
		
		graphics->updateRobotGraphics("HRP4C0", redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[0]));
		
		// graphics->updateRobotGraphics("HRP4C0", redis_client.getEigen("robot_q"));
		// graphics->updateRobotGraphics("HRP4C0", redis_client.getEigen("robot_qi") + redis_client.getEigen("robot_dq")); // + redis_client.getEigen("robot_dq")

		graphics->updateRobotGraphics("HRP4C1", redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[1]));

		if (index == 8) {
			changed_recently = false; // better logic for this
		}
		
		// Eigen::Vector3d ra_end_effector_pos = redis_client.getEigen("1::11::pos");
		// Eigen::Vector3d la_end_effector_pos = redis_client.getEigen("1::23::pos");

		// // Calculate the L2 norm of the difference between the end effectors
		// double l2_norm = (ra_end_effector_pos - la_end_effector_pos).norm();

		// double last_background_change_time = 0.0; // Initialize the last background change time

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

		// // if (l2_norm < CLAP_THRESHOLD) {
		// // 	if (!changed_recently) {
		// // 		int current_clap_count = redis_client.getInt(CLAP_KEY);
		// // 		redis_client.setInt(CLAP_KEY, current_clap_count + 1);

		// // 		int background_index = (current_clap_count + 1) % background_paths.size();
		// // 		if (IMAGE_BACKGROUND_MODE) {
		// // 			setBackgroundImage(graphics, background_paths[background_index]);
		// // 		}
		// // 		// Update the last background change time
		// // 		last_background_change_time = time;
		// // 		changed_recently = true;
		// // 	}
		// // }

		// if (l2_norm < CLAP_THRESHOLD) {
		// 	//graphics->setBackgroundColor(1.0, 1.0, 1.0);  // White RGB
		// }

		// // Retrieve the head position
		// Eigen::Vector3d head_pos = redis_client.getEigen("1::5::pos");

		// // Check if both hands are above the head
		// bool hands_above_head = (head_pos.z() - ra_end_effector_pos.z() < HANDS_ABOVE_HEAD_THRESHOLD ||
		// 						head_pos.z() - la_end_effector_pos.z() < HANDS_ABOVE_HEAD_THRESHOLD);
		// redis_client.setBool(HANDS_ABOVE_HEAD_KEY, hands_above_head);

		// if (hands_above_head) {
		// 	//graphics->setBackgroundColor(0.0, 0.0, 0.0);  // Black RGB
		// }

		// double hands_above = head_pos.z() - ra_end_effector_pos.z();
		// //std::cout << hands_above << std::endl;
		
		// //std::cout << head_pos.z() - ra_end_effector_pos.z() << std::endl;
		// // Set transparency for odd-numbered robots if hands are above the head
		// //if (hands_above < 0.1) {
		// // if (true) {
		// // 	graphics->showTransparency(true, "toro0", 0.0);
		// // 	graphics->showTransparency(true, "toro1", 0.0);
		// // 	graphics->showTransparency(true, "toro2", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro3", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro4", 0.0);
		// // 	graphics->showTransparency(true, "toro5", 1.0);
		// // 	graphics->showTransparency(true, "toro6", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro7", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro8", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro9", 0.0); // Fully transparent
		// // } 
		// // else if (0.3 > hands_above > 0.1) {
		// // 	graphics->showTransparency(true, "toro0", 0.0);
		// // 	graphics->showTransparency(true, "toro1", 0.0);
		// // 	graphics->showTransparency(true, "toro2", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro3", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro4", 1.0);
		// // 	graphics->showTransparency(true, "toro4", 1.0);
		// // 	graphics->showTransparency(true, "toro5", 1.0);
		// // 	graphics->showTransparency(true, "toro6", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro7", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro8", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro9", 0.0); // Fully transparent
		// // } 
		// // else if (0.6 > hands_above > 0.3) {
		// // 	graphics->showTransparency(true, "toro0", 1.0);
		// // 	graphics->showTransparency(true, "toro1", 0.0);
		// // 	graphics->showTransparency(true, "toro2", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro3", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro4", 1.0);
		// // 	graphics->showTransparency(true, "toro5", 0.0);
		// // 	graphics->showTransparency(true, "toro6", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro7", 0.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro8", 1.0); // Fully transparent
		// // 	graphics->showTransparency(true, "toro9", 0.0); 
		// // }
		// // else {
		// // 	graphics->showTransparency(true, "toro0", 1.0);
		// // 	graphics->showTransparency(true, "toro1", 1.0);
		// // 	graphics->showTransparency(true, "toro2", 1.0); 
		// // 	graphics->showTransparency(true, "toro3", 1.0); 
		// // 	graphics->showTransparency(true, "toro4", 1.0);
		// // 	graphics->showTransparency(true, "toro5", 1.0);
		// // 	graphics->showTransparency(true, "toro6", 1.0); 
		// // 	graphics->showTransparency(true, "toro7", 1.0);
		// // 	graphics->showTransparency(true, "toro8", 1.0); 
		// // 	graphics->showTransparency(true, "toro9", 1.0); 
		// // }
    
        // {
		// 	lock_guard<mutex> lock(mutex_update);
			
		// }

		// if (redis_client.getBool(GLOBAL_CAM) && conmove) {
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
		// } else if (redis_client.getBool(GLOBAL_CAM) && !conmove) {

		// } else {
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
			
		// 	hannah_ui_torques = graphics->getUITorques(hannah_name);
		// 	tracy_ui_torques = graphics->getUITorques(tracy_name);
		// }
	}

    // stop simulation
	fSimulationRunning = false;
	sim_thread.join();
	compute_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim,
				const std::vector<double>& lower_limit,
				const std::vector<double>& upper_limit) {
    // fSimulationRunning = true;

	// initialize timer 
	auto t1 = high_resolution_clock::now();
	auto t2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = t2 - t1;

    // create redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

	// setup loop variables
	int ROBOT_DOF = 38;
	VectorXd hannah_control_torques = VectorXd::Zero(ROBOT_DOF);
	VectorXd tracy_control_torques = VectorXd::Zero(ROBOT_DOF);
	bool flag_reset = false;

	VectorXd hannah_robot_q = redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[0]);
	VectorXd hannah_robot_dq = redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[0]);
	VectorXd tracy_robot_q = redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[1]);
	VectorXd tracy_robot_dq = redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[1]);

	// create redis client get and set pipeline 
	redis_client.addToReceiveGroup(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY, hannah_control_torques);
	redis_client.addToReceiveGroup(TRACY_TORO_JOINT_TORQUES_COMMANDED_KEY, tracy_control_torques);

	redis_client.addToSendGroup(MULTI_TORO_JOINT_ANGLES_KEY[0], hannah_robot_q);
	redis_client.addToSendGroup(MULTI_TORO_JOINT_ANGLES_KEY[1], tracy_robot_q);
	redis_client.addToSendGroup(MULTI_TORO_JOINT_VELOCITIES_KEY[0], hannah_robot_dq);
	redis_client.addToSendGroup(MULTI_TORO_JOINT_VELOCITIES_KEY[1], tracy_robot_dq);

    // create a timer
    double sim_freq = 2000;
    SaiCommon::LoopTimer timer(sim_freq);

    sim->setTimestep(1.0 / sim_freq);
    sim->enableGravityCompensation(false);

    sim->disableJointLimits(hannah_name);
	sim->disableJointLimits(tracy_name);

    while (fSimulationRunning) {
        timer.waitForNextLoop();
        const double time = timer.elapsedSimTime();

		// execute read callback 
		redis_client.receiveAllFromGroup();

		// query reset key 
		flag_reset = redis_client.getBool(RESET_SIM_KEY);  
		if (flag_reset) {
			// sim->resetWorld(world_file);
			sim->setJointPositions(hannah_name, hannah_q_desired);
			sim->setJointPositions(tracy_name, tracy_q_desired);
			sim->setJointVelocities(hannah_name, 0 * hannah_q_desired);
			sim->setJointVelocities(tracy_name, 0 * tracy_q_desired);
			// redis_client.set(RESET_SIM_KEY, "0");
			redis_client.set(MULTI_RESET_CONTROLLER_KEY[0], "1");  // hannah
			redis_client.set(MULTI_RESET_CONTROLLER_KEY[1], "1");  // tracy 

			// reset joint angles and velocities in the keys 
			redis_client.setEigen(HANNAH_TORO_JOINT_ANGLES_KEY, hannah->q()); 
			redis_client.setEigen(HANNAH_TORO_JOINT_VELOCITIES_KEY, hannah->dq()); 
			redis_client.setEigen(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY, 0 * hannah->q());

			redis_client.setEigen(TRACY_TORO_JOINT_ANGLES_KEY, tracy->q()); 
			redis_client.setEigen(TRACY_TORO_JOINT_VELOCITIES_KEY, tracy->dq()); 
			redis_client.setEigen(TRACY_TORO_JOINT_TORQUES_COMMANDED_KEY, 0 * tracy->q());

			hannah_control_torques.setZero();
			tracy_control_torques.setZero();

		} 

		// t1 = high_resolution_clock::now();
        // VectorXd hannah_control_torques = redis_client.getEigen(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY);
		// VectorXd tracy_control_torques = redis_client.getEigen(TRACY_TORO_JOINT_TORQUES_COMMANDED_KEY);
		// t2 = high_resolution_clock::now();
		// ms_double = t2 - t1;
		// std::cout << "get torques time: " << ms_double.count() << " ms\n";

		// std::cout << flag_reset << "\n";
		// std::cout << hannah_control_torques.transpose() << "\n";

        // {
            // lock_guard<mutex> lock(mutex_torques);
            sim->setJointTorques(hannah_name, hannah_control_torques);
			sim->setJointTorques(tracy_name, tracy_control_torques);
        // }

		// t1 = high_resolution_clock::now();
		// if (!flag_reset) {
        	sim->integrate();
		// }
		// t2 = high_resolution_clock::now();
		// ms_double = t2 - t1;
		// std::cout << "integrate time: " << ms_double.count() << " ms\n";

		hannah_robot_q = sim->getJointPositions(hannah_name);
    	hannah_robot_dq = sim->getJointVelocities(hannah_name);
		tracy_robot_q = sim->getJointPositions(tracy_name);
    	tracy_robot_dq = sim->getJointVelocities(tracy_name);
		
		// hannah->setQ(hannah_robot_q);
		// hannah->updateModel();
		// tracy->setQ(tracy_robot_q);
		// tracy->updateModel();

		// std::vector<int> right_leg_joints = {6, 7, 8, 9, 10, 11};
		// std::vector<int> left_leg_joints = {12, 13, 14, 15, 16, 17};
		// std::vector<int> torso_joints = {18, 19};
		// std::vector<int> right_arm_joints = {20, 21, 22, 23, 24, 25, 26};
		// std::vector<int> left_arm_joints = {27, 28, 29, 30, 31, 32, 33};
		// std::vector<int> head_joints = {34, 35};

		// std::vector<int> limited_joints = {6, 7, 8, 9, 12, 13, 14, 15};
		// std::vector<int> limited_joints = {9, 15, 6, 7, 9, 10, 11, 12, 13, 16, 17, 23, 30, 19};
		// std::vector<int> limited_joints = {9, 15, 6, 7, 9, 10, 11, 12, 13, 16, 17};

		// std::vector<int> limited_joints = {6, 7, 8, 12, 13, 14, 23, 30};
		// std::vector<int> limited_joints = {6, 7, 8, 9, 12, 13, 14, 15};
		// std::vector<int> limited_joints = {9, 15, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19};
		// limited_joints.push_back(20);
		// limited_joints.push_back(21);
		// limited_joints.push_back(27);
		// limited_joints.push_back(28);
		// std::vector<int> limited_joints;
		// for (int i = 0; i < robot_q.size(); ++i) {
		// 	limited_joints.push_back(i);
		// }

		for (auto id : limited_joints) {
			if (hannah_robot_q(id) > upper_limit[id]) {
				hannah_robot_q[id] = upper_limit[id];
				// hannah_robot_dq(id) = 0;
			} else if (hannah_robot_q(id) < lower_limit[id]) {
				hannah_robot_q(id) = lower_limit[id];
				// hannah_robot_dq(id) = 0;
			}
		}

		for (auto id : limited_joints) {
			if (tracy_robot_q(id) > upper_limit[id]) {
				tracy_robot_q[id] = upper_limit[id];
				// tracy_robot_dq(id) = 0;
			} else if (tracy_robot_q(id) < lower_limit[id]) {
				tracy_robot_q(id) = lower_limit[id];
				// tracy_robot_dq(id) = 0;
			}
		}

		// hannah->setQ(hannah_robot_q);
		// hannah->setDq(hannah_robot_dq);
		// hannah->updateModel();

		// tracy->setQ(tracy_robot_q);
		// tracy->setDq(tracy_robot_dq);
		// tracy->updateModel();

		// VectorXd robot_q = (hannah_robot_q + tracy_robot_q)/2;
		// VectorXd robot_dq = (hannah_robot_dq + tracy_robot_dq)/2;

		// redis_client.setEigen(MULTI_TORO_JOINT_ANGLES_KEY[0], hannah_robot_q);
		// redis_client.setEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[0], hannah_robot_dq);
		// redis_client.setEigen(MULTI_TORO_JOINT_ANGLES_KEY[1], tracy_robot_q);
		// redis_client.setEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[1], tracy_robot_dq);

        //robot_dq = 0.1 * VectorXd::Ones(robot_dq.size()) * sin(time);

        // // Get the mass matrix
        // MatrixXd hannah_robot_M = hannah->M();
        // VectorXd hannah_g = hannah->jointGravityVector();
        // double hannah_kinetic_energy = 0.5 * hannah_robot_dq.transpose() * hannah_robot_M * hannah_robot_dq;
		// Eigen::VectorXd hanah_comPosition = hannah->comPosition();
		// double hannah_potential_energy = 90 * hanah_comPosition(2) * -9.8;
        // double hannah_lagrangian = hannah_kinetic_energy - hannah_potential_energy;
        
		// redis_client.set(HANNAH_KINETIC, std::to_string(hannah_kinetic_energy));
		// redis_client.set(HANNAH_POTENTIAL, std::to_string(hannah_potential_energy));
		// redis_client.set(HANNAH_LAGRANGIAN, std::to_string(hannah_lagrangian));

		// MatrixXd tracy_robot_M = tracy->M();
        // VectorXd tracy_g = tracy->jointGravityVector();
        // double tracy_kinetic_energy = 0.5 * tracy_robot_dq.transpose() * tracy_robot_M * tracy_robot_dq;
		// //Eigen::VectorXd tracy_comPosition = tracy->comPosition();
		// //double tracy_potential_energy = 90 * tracy_comPosition(2) * -9.8;
        // double tracy_lagrangian = tracy_kinetic_energy - tracy_potential_energy;
        
		// redis_client.set(TRACY_KINETIC, std::to_string(tracy_kinetic_energy));
		// redis_client.set(TRACY_POTENTIAL, std::to_string(tracy_potential_energy));
		// redis_client.set(TRACY_LAGRANGIAN, std::to_string(tracy_lagrangian));

        // redis_client.setEigen(HANNAH_TORO_JOINT_ANGLES_KEY, hannah_robot_q);
        // redis_client.setEigen(HANNAH_TORO_JOINT_VELOCITIES_KEY, hannah_robot_dq);
		// redis_client.setEigen(TRACY_TORO_JOINT_ANGLES_KEY, tracy_robot_q);
        // redis_client.setEigen(TRACY_TORO_JOINT_VELOCITIES_KEY, tracy_robot_dq);

        // {
        //     lock_guard<mutex> lock(mutex_update);
        // }


		// execute write callback
		redis_client.sendAllFromGroup();

    }
    timer.stop();
    cout << "\nSimulation loop timer stats:\n";
    timer.printInfoPostRun();
}

/**
 * Separate thread to compute:
 * 	- robot vs. human kinetic energies
 *  - robot vs. human energy consumption 
 */
void computeEnergy() {
	auto redis_client_energy = SaiCommon::RedisClient();
	redis_client_energy.connect();
	auto hannah = std::make_shared<SaiModel::SaiModel>(toro_file, false);
	auto tracy = std::make_shared<SaiModel::SaiModel>(toro_file, false);

	auto human_hannah = std::make_shared<SaiModel::SaiModel>(human_file, false);
	auto human_tracy = std::make_shared<SaiModel::SaiModel>(human_file, false);

	// gear inertia scaling for mass matrix diagonal
	double motor_inertia = 0.1;
	double gear_inertia = 20;
	MatrixXd motor_inertia_bias = motor_inertia * gear_inertia * gear_inertia * MatrixXd::Identity(hannah->dof(), hannah->dof());	

    // create a timer
    double sim_freq = 100;
    SaiCommon::LoopTimer timer(sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();

		/*
			Hannah robot 
		*/
		hannah->setQ(redis_client_energy.getEigen(HANNAH_TORO_JOINT_ANGLES_KEY));
		hannah->setDq(redis_client_energy.getEigen(HANNAH_TORO_JOINT_VELOCITIES_KEY));
		hannah->updateModel();

		double hannah_kinetic_energy = 0.5 * hannah->dq().transpose() * (hannah->M() + 0 * motor_inertia_bias) * hannah->dq();  // robot kinetic energy WITH ADDED MOTOR INERTIA BIAS

		// robot effort from joint torques: may need to include the motor inertia through the gear ratio 
		VectorXd hannah_control_torques = redis_client_energy.getEigen(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY);
		double hannah_robot_power = hannah_control_torques.cwiseAbs().transpose() * hannah->dq().cwiseAbs();

		/*
			Tracy robot 
		*/
		tracy->setQ(redis_client_energy.getEigen(TRACY_TORO_JOINT_ANGLES_KEY));
		tracy->setDq(redis_client_energy.getEigen(TRACY_TORO_JOINT_VELOCITIES_KEY));
		tracy->updateModel();

		double tracy_kinetic_energy = 0.5 * tracy->dq().transpose() * (tracy->M() + 0 * motor_inertia_bias) * tracy->dq();  // robot kinetic energy WITH ADDED MOTOR INERTIA BIAS

		// robot effort from joint torques: may need to include the motor inertia through the gear ratio 
		VectorXd tracy_control_torques = redis_client_energy.getEigen(TRACY_TORO_JOINT_TORQUES_COMMANDED_KEY);
		double tracy_robot_power = tracy_control_torques.cwiseAbs().transpose() * tracy->dq().cwiseAbs();

		/*
			Human Hannah 
		*/
		human_hannah->setQ(redis_client_energy.getEigen(HANNAH_TORO_JOINT_ANGLES_KEY));
		human_hannah->setDq(redis_client_energy.getEigen(HANNAH_TORO_JOINT_VELOCITIES_KEY));
		human_hannah->updateModel();

		double hannah_human_kinetic_energy = 0.5 * human_hannah->dq().transpose() * human_hannah->M() * human_hannah->dq();

		// human effort estimated from a similar robot urdf, but with human values 
		VectorXd unit_mass_torques = (hannah->M() + motor_inertia_bias).inverse() * hannah_control_torques;
		double hannah_human_power = human_hannah->dq().cwiseAbs().transpose() * (human_hannah->M() * unit_mass_torques).cwiseAbs();

		/*
			Human Tracy 
		*/
		human_tracy->setQ(redis_client_energy.getEigen(TRACY_TORO_JOINT_ANGLES_KEY));
		human_tracy->setDq(redis_client_energy.getEigen(TRACY_TORO_JOINT_VELOCITIES_KEY));
		human_tracy->updateModel();

		double tracy_human_kinetic_energy = 0.5 * human_tracy->dq().transpose() * human_tracy->M() * human_tracy->dq();

		// human effort estimated from a similar robot urdf, but with human values 
		unit_mass_torques = (tracy->M() + motor_inertia_bias).inverse() * tracy_control_torques;
		double tracy_human_power = human_tracy->dq().cwiseAbs().transpose() * (human_tracy->M() * unit_mass_torques).cwiseAbs();

		std::cout << "robot power: " << hannah_robot_power << "\n";
		// std::cout << "tracy human power: " << tracy_human_power << "\n";

		// publish data
		redis_client_energy.setDouble(HANNAH_ROBOT_KINETIC_KEY, hannah_kinetic_energy);
		redis_client_energy.setDouble(TRACY_ROBOT_KINETIC_KEY, tracy_kinetic_energy);
		redis_client_energy.setDouble(HANNAH_HUMAN_KINETIC_KEY, hannah_human_kinetic_energy);
		redis_client_energy.setDouble(TRACY_HUMAN_KINETIC_KEY, tracy_human_kinetic_energy);

		redis_client_energy.setDouble(HANNAH_ROBOT_EFFORT_KEY, hannah_robot_power);
		redis_client_energy.setDouble(TRACY_ROBOT_EFFORT_KEY, tracy_robot_power);
		redis_client_energy.setDouble(HANNAH_HUMAN_EFFORT_KEY, hannah_human_power);
		redis_client_energy.setDouble(TRACY_HUMAN_EFFORT_KEY, tracy_human_power);
	}
}