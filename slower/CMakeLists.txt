cmake_minimum_required(VERSION 3.5)
project(my_project_name)

find_package(benchmark REQUIRED)
find_package(autodiff REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

add_executable(bench_demo src/traj_opt.cpp)

target_include_directories(bench_demo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${EIGEN3_INCLUDE_DIR}")

target_compile_features(bench_demo PUBLIC cxx_std_20)

target_link_libraries(bench_demo PUBLIC benchmark::benchmark autodiff::autodiff Eigen3::Eigen TBB::tbb)
