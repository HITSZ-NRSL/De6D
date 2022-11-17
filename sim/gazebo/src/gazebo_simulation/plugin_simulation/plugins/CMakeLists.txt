cmake_minimum_required(VERSION 2.8.3)
project(plugins)
set(CMAKE_CXX_STANDARD 17)
#set( CMAKE_BUILD_TYPE "release" )
find_package(gazebo REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        visualization_msgs
        )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${GAZEBO_INCLUDE_DIRS}
)
link_directories(
        ${GAZEBO_LIBRARY_DIRS}
)
catkin_package(
        INCLUDE_DIRS ${GAZEBO_INCLUDE_DIRS}
        LIBRARIES
        CATKIN_DEPENDS roscpp  visualization_msgs
)
message(${GAZEBO_LIBRARIES})
add_executable(gz_marker src/gz_marker.cpp)
target_link_libraries(gz_marker
        ${catkin_LIBRARIES}
        ${GAZEBO_LIBRARIES}
        )