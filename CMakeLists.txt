cmake_minimum_required(VERSION 3.13)

project(PimBloomFilters VERSION 0.1)

set(CMAKE_BUILD_TYPE Debug) 
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)

add_executable(unit_test1 unit_test1.cpp)

execute_process(COMMAND dpu-pkg-config --cflags dpu OUTPUT_VARIABLE DPU_INCLUDE)
string(REPLACE "-I" "" DPU_INCLUDE ${DPU_INCLUDE})
string(REPLACE "\n" "" DPU_INCLUDE ${DPU_INCLUDE})

execute_process(COMMAND dpu-pkg-config --libs dpu OUTPUT_VARIABLE DPU_LINK)
string(REPLACE "\n" "" DPU_LINK ${DPU_LINK})

target_include_directories(unit_test1 PUBLIC "." ${DPU_INCLUDE})
target_link_libraries(unit_test1 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(unit_test1 PRIVATE ${DPU_LINK})