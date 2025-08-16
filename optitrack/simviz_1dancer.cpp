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
 std::shared_ptr<SaiGraphics::SaiGraphics> graphics;
 
 #include "redis_keys.h"
 
 using namespace Eigen;
 using namespace std;
 
 // mutex and globals
 VectorXd hannah_ui_torques;
 
 mutex mutex_torques, mutex_update;
 
 // specify urdf and robots 
 static const string hannah_name = "HRP4C0";
 static const string camera_name = "camera_fixed";
 const std::string yaml_fname = "./resources/controller_settings_multi_dancers.yaml";
 
 static const string toro_file = "./resources/model/HRP4c.urdf";
 static const string human_file = "./resources/model/human.urdf";
 static const string world_file = "./resources/world/world_basic_1.urdf";
 
 bool DEBUG = false;
 std::vector<int> limited_joints;
 VectorXd hannah_q_desired(38);
 
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
     hannah_ui_torques = VectorXd::Zero(hannah->dof());
     hannah_q_desired << 0, 0.75, 0, 0, 0, 0, 
                     0, -0.1, -0.25, 0.5, -0.25, 0, 0.1, 
                     0, 0.1, -0.25, 0.5, -0.25, 0, -0.1, 
                     0, 0,
                     -0.1, -0.2, 0.3, -1.3, 0.2, 0.7, -0.7, 
                     -0.1, 0.2, -0.3, -1.3, 0.7, 0.7, -0.7, 
                     0, 0;
 
     hannah->setQ(hannah_q_desired);
     hannah->updateModel();
 
     auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
     
     sim->setJointPositions(hannah_name, hannah->q());
     sim->setJointVelocities(hannah_name, hannah->dq());
 
 
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
 
     // start simulation thread
     thread sim_thread(simulation, sim, lower_joint_limits, upper_joint_limits);
     thread compute_thread(computeEnergy);
     
     int robot_index = 0; // index to track which robot to update next
 
     SaiCommon::LoopTimer timer(120, 1e6);
     timer.reinitializeTimer(1e9);
 
 
     // while window is open:
     while (graphics->isWindowOpen() && fSimulationRunning) {
         timer.waitForNextLoop();
         graphics->setBackgroundColor(1.0, 1.0, 1.0);
        //  graphics->updateRobotGraphics("HRP4C0", redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[0]));
         
         // graphics->updateRobotGraphics("HRP4C0", redis_client.getEigen("robot_q"));
         graphics->updateRobotGraphics("HRP4C0", redis_client.getEigen("robot_qi") + redis_client.getEigen("robot_dq")); // + redis_client.getEigen("robot_dq")
 


         graphics->renderGraphicsWorld();
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
     bool flag_reset = false;
 
     VectorXd hannah_robot_q = redis_client.getEigen(MULTI_TORO_JOINT_ANGLES_KEY[0]);
     VectorXd hannah_robot_dq = redis_client.getEigen(MULTI_TORO_JOINT_VELOCITIES_KEY[0]);
     
     // create redis client get and set pipeline 
     redis_client.addToReceiveGroup(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY, hannah_control_torques);
     
     redis_client.addToSendGroup(MULTI_TORO_JOINT_ANGLES_KEY[0], hannah_robot_q);
     redis_client.addToSendGroup(MULTI_TORO_JOINT_VELOCITIES_KEY[0], hannah_robot_dq);
 
     // create a timer
     double sim_freq = 2000;
     SaiCommon::LoopTimer timer(sim_freq);
 
     sim->setTimestep(1.0 / sim_freq);
     sim->enableGravityCompensation(false);
 
     sim->disableJointLimits(hannah_name);
 
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
             sim->setJointVelocities(hannah_name, 0 * hannah_q_desired);
             // redis_client.set(RESET_SIM_KEY, "0");
             redis_client.set(MULTI_RESET_CONTROLLER_KEY[0], "1");  // hannah
 
             // reset joint angles and velocities in the keys 
             redis_client.setEigen(HANNAH_TORO_JOINT_ANGLES_KEY, hannah->q()); 
             redis_client.setEigen(HANNAH_TORO_JOINT_VELOCITIES_KEY, hannah->dq()); 
             redis_client.setEigen(HANNAH_TORO_JOINT_TORQUES_COMMANDED_KEY, 0 * hannah->q());
 
             
             hannah_control_torques.setZero();
 
         } 
 

        sim->setJointTorques(hannah_name, hannah_control_torques);
        sim->integrate();
 
         hannah_robot_q = sim->getJointPositions(hannah_name);
         hannah_robot_dq = sim->getJointVelocities(hannah_name);
         
 
         for (auto id : limited_joints) {
             if (hannah_robot_q(id) > upper_limit[id]) {
                 hannah_robot_q[id] = upper_limit[id];
                 // hannah_robot_dq(id) = 0;
             } else if (hannah_robot_q(id) < lower_limit[id]) {
                 hannah_robot_q(id) = lower_limit[id];
                 // hannah_robot_dq(id) = 0;
             }
         }
 
 
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
 #include <iomanip>  // for std::setprecision

 void computeEnergy() {
     // Redis client
     auto redis_client = SaiCommon::RedisClient();
     redis_client.connect();
 
     // Load robot and human models
     auto hannah       = std::make_shared<SaiModel::SaiModel>(toro_file,  false);
     auto human_hannah = std::make_shared<SaiModel::SaiModel>(human_file, false);
 
     // (Optional) motor/gear inertia bias
     double motor_inertia = 0.1;
     double gear_inertia  = 20.0;
     MatrixXd motor_bias  = motor_inertia * gear_inertia * gear_inertia
                          * MatrixXd::Identity(hannah->dof(), hannah->dof());
 
     // Timer at 100 Hz
     const double sim_freq = 100.0;
     SaiCommon::LoopTimer timer(sim_freq);
 
     // Accumulators for ΔKE and Δp
     double prev_ke         = 0.0;
     VectorXd prev_p        = VectorXd::Zero(hannah->dof());
     double sum_delta_ke    = 0.0;
     double sum_delta_p     = 0.0;
     size_t sample_count    = 0;
     double last_print_time = 0.0;
 
     while (fSimulationRunning) {
         timer.waitForNextLoop();
         double t = timer.elapsedSimTime();
 
         // --- Robot kinetic energy & momentum ---
         VectorXd q   = redis_client.getEigen("robot_qi") + redis_client.getEigen("robot_dq");
         VectorXd dq  = redis_client.getEigen("robot_dq");
         hannah->setQ(q);
         hannah->setDq(dq);
         hannah->updateModel();

        VectorXd hannah_control_torques = redis_client.getEigen("robot_qi") + redis_client.getEigen("robot_dq"); 
		double hannah_robot_power = hannah_control_torques.cwiseAbs().transpose() * hannah->dq().cwiseAbs();
        std::cout<<"POWER: "<<hannah_robot_power<<std::endl;
 
         MatrixXd M = hannah->M() + MatrixXd::Zero(hannah->dof(), hannah->dof()); // + motor_bias if desired
 
         // dot() form yields a double
         double ke_current = 0.5 * dq.dot(M * dq);
         VectorXd p_current = M * dq;
 
         // instantaneous changes
         double delta_ke     = std::abs(ke_current - prev_ke);
         double delta_p_norm = (p_current - prev_p).norm();
 
         // accumulate
         sum_delta_ke += delta_ke;
         sum_delta_p  += delta_p_norm;
         ++sample_count;
 
         // update previous
         prev_ke = ke_current;
         prev_p  = p_current;
 
         // print averages once per second
         if (t - last_print_time >= 1.0) {
             double avg_delta_ke = sum_delta_ke / sample_count;
             double avg_delta_p  = sum_delta_p  / sample_count;
             std::cout << std::fixed << std::setprecision(4)
                       << "[t=" << t << "s] avgΔKE: " << avg_delta_ke
                       << ", avgΔp: "  << avg_delta_p  << "\n";
             last_print_time = t;
         }
     }
 }