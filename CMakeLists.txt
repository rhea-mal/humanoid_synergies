cmake_minimum_required (VERSION 3.0)
project (CENTER_DEMO)
if (${CMAKE_SYSTEM_NAME} MATCHES Linux)
	SET(CMAKE_CXX_FLAGS "-std=c++17 -I/usr/include -I/usr/local/include -O3")
elseif (${CMAKE_SYSTEM_NAME} MATCHES Darwin)
	SET(CMAKE_CXX_FLAGS "-std=c++17 -I/usr/include -I/usr/local/include -I/opt/homebrew/include -w -O3")
endif()
# SET(CMAKE_BUILD_TYPE "Debug")  
SET(CMAKE_BUILD_TYPE "Release")  

# set common dependencies
# - eigen3
set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
if(NOT EIGEN3_INCLUDE_DIR)
	find_package(Eigen3 QUIET)
	# TODO: Find automatic way to find for Ubuntu 14.04
	if(NOT EIGEN3_INCLUDE_DIR)
		set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
	endif()
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# include Chai3d 
# -- this is required by Chai3d to pick the correct path for OpenGL headers
FIND_PACKAGE(CHAI3D REQUIRED)
ADD_DEFINITIONS(${CHAI3D_DEFINITIONS})
if (CMAKE_SYSTEM_NAME MATCHES Darwin) 
  ADD_DEFINITIONS(-D_MACOSX -DMACOSX)
elseif (CMAKE_SYSTEM_NAME MATCHES Linux)
  ADD_DEFINITIONS(-D_LINUX -DLINUX)
endif () #NOTE: Windows is currently unsupported
LINK_DIRECTORIES(${CHAI3D_LIBRARY_DIRS})
INCLUDE_DIRECTORIES(${CHAI3D_INCLUDE_DIRS})

# - SAI-SIMULATION
find_package(SAI-SIMULATION REQUIRED)
include_directories(${SAI-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI-SIMULATION_DEFINITIONS})

# - SAI-GRAPHICS
FIND_PACKAGE(SAI-GRAPHICS REQUIRED)
include_directories(${SAI-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI-GRAPHICS_DEFINITIONS})

# - SAI-COMMON
FIND_PACKAGE(SAI-COMMON REQUIRED)
include_directories(${SAI-COMMON_INCLUDE_DIRS})

# - SAI-URDF
FIND_PACKAGE(SAI-URDF REQUIRED)
include_directories(${SAI-URDF_INCLUDE_DIRS})

# - SAI-MODEL
find_package(SAI-MODEL REQUIRED)
include_directories(${SAI-MODEL_INCLUDE_DIRS})
add_definitions(${SAI-MODEL_DEFINITIONS})

# - SAI-PRIMITIVES
find_package(SAI-PRIMITIVES REQUIRED)
include_directories(${SAI-PRIMITIVES_INCLUDE_DIRS})
add_definitions(${SAI-PRIMITIVES_DEFINITIONS})

# - SAI-INTERFACES
find_package(SAI-INTERFACES REQUIRED)
include_directories(${SAI-INTERFACES_INCLUDE_DIRS})
add_definitions(${SAI-INTERFACES_DEFINITIONS})

# - hiredis
find_library(HIREDIS_LIBRARY hiredis)

# - yaml 
# find yaml-cpp
find_package(yaml-cpp REQUIRED)

# - library
add_library(center-demo STATIC ${PROJECT_SOURCE_DIR}/include/Human.cpp)

# define libraries for linking 
set(CENTER_DEMO_COMMON_LIBRARIES
    ${SAI-INTERFACES_LIBRARIES}
    ${SAI-PRIMITIVES_LIBRARIES}
    ${SAI-SIMULATION_LIBRARIES}
    ${SAI-GRAPHICS_LIBRARIES}
    ${SAI-MODEL_LIBRARIES}
    ${SAI-URDF_LIBRARIES}
    ${SAI-COMMON_LIBRARIES}
    ${CHAI3D_LIBRARIES}
    ${HIREDIS_LIBRARY}
    center-demo
    yaml-cpp::yaml-cpp
	)

set (CENTER_DEMO_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(optitrack)