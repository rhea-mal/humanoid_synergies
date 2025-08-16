# Humanoid Postural Synergies

This repository contains the code, data, and documentation for the paper "**Humanoid Synergy Editor Leveraging Postural Synergies for Kinematically Optimal Free-Space Control**" (Humanoids 2025).

The project introduces **SynSculptor**, a humanoid motion analysis and editing framework leveraging postural synergies for training-free human-like motion scripting.

## Key Features
- **MotionGPT Fine-tuning**: Enhances humanoid control using synergy-augmented MotionGPT for better realism and kinematic accuracy.
- **Real-Time Mapping**: Maps OptiTrack motion capture data to a humanoid robot (HRP4c).
- **Energetic Efficiency**: Evaluates energy consumption and foot-sliding behavior for various motion sequences.

## Project Structure
- **`CMakeLists.txt`**: Build configuration for the project.
- **`bin/`**: Compiled binaries and executables.
- **`include/`**: Header files for core libraries.
- **`optitrack/`**: Data related to motion capture.
- **`MotionGPT/`**: Implementation for synergy-augmented MotionGPT.
- **`drivers/`**: Drivers for external systems (OptiTrack, HRP4c).
- **`model/`**: Humanoid robot models and configurations.
- **`synergies/`**: Code for postural synergy extraction.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rhea-mal/humanoid_synergies.git
    cd humanoid_synergies
    ```

2. Follow directions to install OpenSAI
    
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Build the project:
    ```bash
    mkdir build
    cd build
    cmake .. 
    make
    ```

