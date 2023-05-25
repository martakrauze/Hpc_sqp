cmake_minimum_required(VERSION 3.5)
project(Trajectory_optimization)

find_package(autodiff REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(traj_opt src/traj_opt.cpp)

target_include_directories(traj_opt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${EIGEN3_INCLUDE_DIR}")

target_compile_features(traj_opt PUBLIC cxx_std_20)

target_link_libraries(traj_opt PUBLIC autodiff::autodiff Eigen3::Eigen)
