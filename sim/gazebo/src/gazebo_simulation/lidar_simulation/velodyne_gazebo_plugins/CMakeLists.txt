cmake_minimum_required(VERSION 3.0.2)
project(velodyne_gazebo_plugins)
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
set( CMAKE_BUILD_TYPE "release" )  #release debug 模式
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  tf
  gazebo_ros
)
find_package(gazebo REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

catkin_package(
  INCLUDE_DIRS include ${GAZEBO_INCLUDE_DIRS}
  LIBRARIES gazebo_ros_velodyne_laser gazebo_ros_velodyne_gpu_laser
  CATKIN_DEPENDS roscpp sensor_msgs gazebo_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

add_library(gazebo_ros_velodyne_laser src/GazeboRosVelodyneLaser.cpp)
target_link_libraries(gazebo_ros_velodyne_laser
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  RayPlugin
)

add_library(gazebo_ros_velodyne_gpu_laser src/GazeboRosVelodyneLaser.cpp)
target_link_libraries(gazebo_ros_velodyne_gpu_laser
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  GpuRayPlugin
)
target_compile_definitions(gazebo_ros_velodyne_gpu_laser PRIVATE GAZEBO_GPU_RAY=1)
############################ lidar utils #######################################
add_library(gazebo_ros_lidar_utils src/gazebo_ros_lidar_utils.cpp)
target_link_libraries(gazebo_ros_lidar_utils
        ${catkin_LIBRARIES}
        ${GAZEBO_LIBRARIES}
        )

############################ my costom lidar plugin ############################

add_library(gazebo_ros_customized_lidar src/GazeboROSCostomizedLiDAR.cpp)
target_link_libraries(gazebo_ros_customized_lidar
        ${catkin_LIBRARIES}
        ${GAZEBO_LIBRARIES}
        RayPlugin
        )

add_library(gazebo_ros_customized_gpu_lidar src/GazeboROSCostomizedLiDAR.cpp)
target_link_libraries(gazebo_ros_customized_gpu_lidar
        ${catkin_LIBRARIES}
        ${GAZEBO_LIBRARIES}
        GpuRayPlugin
        )
target_compile_definitions(gazebo_ros_customized_gpu_lidar PRIVATE GAZEBO_GPU_RAY=1)
############################ my costom lidar plugin ############################

add_library(gazebo_ros_triggered_lidar src/gazebo_ros_triggered_lidar.cpp)
target_link_libraries(gazebo_ros_customized_lidar
        ${catkin_LIBRARIES}
        ${GAZEBO_LIBRARIES}
        RayPlugin
        )

add_library(gazebo_ros_triggered_gpu_lidar src/gazebo_ros_triggered_lidar.cpp)
target_link_libraries(gazebo_ros_customized_gpu_lidar
        ${catkin_LIBRARIES}
        ${GAZEBO_LIBRARIES}
        GpuRayPlugin
        )
target_compile_definitions(gazebo_ros_triggered_gpu_lidar PRIVATE GAZEBO_GPU_RAY=1)
############################ my costom lidar plugin ############################



install(TARGETS gazebo_ros_velodyne_laser gazebo_ros_velodyne_gpu_laser
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


