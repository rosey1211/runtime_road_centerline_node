cmake_minimum_required(VERSION 3.14)
project(road_centerline)

# C++17 required by rclcpp and LibTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find dependencies ──────────────────────────────────────────────────────────
find_package(ament_cmake      REQUIRED)
find_package(rclcpp            REQUIRED)
find_package(sensor_msgs       REQUIRED)
find_package(geometry_msgs     REQUIRED)
find_package(std_msgs          REQUIRED)
find_package(cv_bridge         REQUIRED)
find_package(control_interfaces REQUIRED)
find_package(OpenCV            REQUIRED)
find_package(yaml-cpp          REQUIRED)

# LibTorch — set TORCH_DIR to your LibTorch installation if not on PATH.
# On Jetson (JetPack 6): export TORCH_DIR=/usr/local/lib/python3.10/dist-packages/torch
# On desktop:            pip show torch -> Location, then append /torch/share/cmake/Torch
if(DEFINED ENV{TORCH_DIR})
  list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_DIR})
endif()
find_package(Torch REQUIRED)

# ── Custom message generation ──────────────────────────────────────────────────
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CenterlineResult.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# ── Node executable ────────────────────────────────────────────────────────────
add_executable(road_centerline_node src/road_centerline_node.cpp)

target_include_directories(road_centerline_node PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /opt/ros/jazzy/include
)

ament_target_dependencies(road_centerline_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  control_interfaces
)

target_link_libraries(road_centerline_node
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

# Link generated message interfaces
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(road_centerline_node "${cpp_typesupport_target}")

# ── Install ────────────────────────────────────────────────────────────────────
install(TARGETS road_centerline_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  USE_SOURCE_PERMISSIONS)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
