cmake_minimum_required(VERSION 3.25)

set(catkin_DIR /opt/ros/noetic/share/catkin/cmake)
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS roscpp tf sensor_msgs image_transport cv_bridge std_msgs mesh_msgs geometry_msgs)
# include(${catkin_DIR}/all.cmake)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../../bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../../lib)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

set(LIBS
    ORB_SLAM2
    ${catkin_LIBRARIES}
)

message("catkin libraries are:"+${catkin_LIBRARIES})

add_executable(rgbd_ros
    main.cc
)
target_link_libraries(rgbd_ros ${LIBS})