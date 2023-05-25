cmake_minimum_required(VERSION 3.5)
project(my_project_name)

find_package(benchmark REQUIRED)
find_package(autodiff REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

add_executable(traj_opt src/traj_opt.cpp)

target_include_directories(traj_opt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${EIGEN3_INCLUDE_DIR}")

target_compile_features(traj_opt PUBLIC cxx_std_20)

target_link_libraries(traj_opt PUBLIC benchmark::benchmark autodiff::autodiff Eigen3::Eigen TBB::tbb)
