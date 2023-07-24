cmake_minimum_required(VERSION 3.13)

project(PimBloomFilters VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set (CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE Debug) 
set(CMAKE_BUILD_TYPE Release)

option(LOG_DPU "Print log messages from DPUs")

add_subdirectory(src)

set(UTILITY_SCRIPTS dtrace.sh)
foreach(SCRIPT ${UTILITY_SCRIPTS})
    configure_file(${SCRIPT} ${SCRIPT} COPYONLY)
endforeach(SCRIPT ${UTILITY_SCRIPTS})

unset(LOG_DPU CACHE)