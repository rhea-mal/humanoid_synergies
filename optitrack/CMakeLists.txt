# # create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CENTER_DEMO_BINARY_DIR}/optitrack)
ADD_EXECUTABLE (verify_optitrack verify_optitrack.cpp ${CENTER_DEMO_COMMON_SOURCE})
ADD_EXECUTABLE (toro_controller toro_controller.cpp ${CENTER_DEMO_COMMON_SOURCE})
ADD_EXECUTABLE (toro_replay toro_replay.cpp ${CENTER_DEMO_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_stage simviz.cpp ${CENTER_DEMO_COMMON_SOURCE})
# ADD_EXECUTABLE (toro_controller_2dancers toro_controller_2dancers.cpp ${CENTER_DEMO_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_2dancers simviz_2dancers.cpp ${CENTER_DEMO_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_1dancer simviz_1dancer.cpp ${CENTER_DEMO_COMMON_SOURCE})
# ADD_EXECUTABLE (rendering rendering.cpp ${CENTER_DEMO_COMMON_SOURCE})
# ADD_EXECUTABLE (visualization visualization.cpp ${CENTER_DEMO_COMMON_SOURCE})
# ADD_EXECUTABLE (simulation_atlas simulation_atlas.cpp ${CENTER_DEMO_COMMON_SOURCE})
# ADD_EXECUTABLE (trace trace.cpp ${CENTER_DEMO_COMMON_SOURCE})


# # and link the library against the executable
TARGET_LINK_LIBRARIES (toro_controller ${CENTER_DEMO_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (toro_replay ${CENTER_DEMO_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_stage ${CENTER_DEMO_COMMON_LIBRARIES})
# TARGET_LINK_LIBRARIES (toro_controller_2dancers ${CENTER_DEMO_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_2dancers ${CENTER_DEMO_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_1dancer ${CENTER_DEMO_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (verify_optitrack ${CENTER_DEMO_COMMON_LIBRARIES})
# TARGET_LINK_LIBRARIES (rendering ${CENTER_DEMO_COMMON_LIBRARIES})
# TARGET_LINK_LIBRARIES (visualization ${CENTER_DEMO_COMMON_LIBRARIES})
# TARGET_LINK_LIBRARIES (simulation_atlas ${CENTER_DEMO_COMMON_LIBRARIES})
# TARGET_LINK_LIBRARIES (trace ${CENTER_DEMO_COMMON_LIBRARIES})

# # export resources such as model files.
# # NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY world DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY model DESTINATION ${APP_RESOURCE_DIR})
#FILE(COPY background DESTINATION ${APP_RESOURCE_DIR})
#FILE(COPY audio DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY controller_settings_1_dancer.yaml controller_settings_2_dancers.yaml controller_settings_multi_dancers.yaml DESTINATION ${APP_RESOURCE_DIR})