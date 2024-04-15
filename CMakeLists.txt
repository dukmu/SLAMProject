cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "-std=c++17 -faligned-new -Wno-deprecated-declarations -O3 -march=native")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
else()
   if(NOT "${OpenCV_VERSION_MAJOR}" STREQUAL "4")
      message(WARNING "OpenCV 4 is required for dnn module. Object detector will not be available.")
   else()
      message("OpenCV 4 found. Make sure OpenCV is built with dnn module and CUDA.")
      add_definitions(-DUSE_DNN)
   endif()
endif()



find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Protobuf REQUIRED)

# JSON
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(Thirdparty/json)

# Dlib
find_package(dlib REQUIRED)


set(PCL_DIR "/usr/local/share/pcl-1.14")
FIND_PACKAGE(PCL 1.14 REQUIRED)
find_package(VDBFusion REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PROTOBUF_INCLUDE_DIR}
${PCL_INCLUDE_DIRS}
${VDBFusion_INCLUDE_DIRS}
)

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


file(GLOB SRC_FILES "src/*.cc" "src/*.cpp")
message("SRC_FILES: ${SRC_FILES}")
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROTOBUF_LIBRARIES}
nlohmann_json::nlohmann_json
stdc++fs
dlib::dlib
${PCL_LIBRARIES}
VDBFusion::vdbfusion
)



# Build Examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(oa-slam
Examples/oa-slam.cc Examples/dataset.cc)
target_link_libraries(oa-slam ${PROJECT_NAME})

add_subdirectory(Examples/rgbd_ros)