cmake_minimum_required(VERSION 3.13)

project(BloomFilters VERSION 0.1 LANGUAGES C CXX DESCRIPTION "Bloom filters on CPU and UPMEM PIM architectures")

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set (CMAKE_CXX_STANDARD 17)
set(LIB_NAME "BloomFilters")

# ------------------------------------------------------------------------------
#                                  Set options                                  
# ------------------------------------------------------------------------------

# set(CMAKE_BUILD_TYPE Debug) 
set(CMAKE_BUILD_TYPE Release)

option(LOG_DPU "Print log messages from DPUs")
option(FSANITIZE "Compile with fsanitize=address flags")

# ------------------------------------------------------------------------------
#                         Get some DPU info and prepare                         
# ------------------------------------------------------------------------------

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)

execute_process(COMMAND dpu-pkg-config --cflags dpu OUTPUT_VARIABLE DPU_H_INCLUDE)
string(REPLACE "-I" "" DPU_H_INCLUDE ${DPU_H_INCLUDE})
string(REPLACE "\n" "" DPU_H_INCLUDE ${DPU_H_INCLUDE})

execute_process(COMMAND dpu-pkg-config --libs dpu OUTPUT_VARIABLE DPU_LINK)
string(REPLACE "\n" "" DPU_LINK ${DPU_LINK})

function(compile_link TARGET)
    target_include_directories(${TARGET} PRIVATE "src" ${DPU_H_INCLUDE})
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra)
    if(FSANITIZE)
        target_compile_options(${TARGET} PRIVATE -fsanitize=leak)
    endif(FSANITIZE)
    if (LOG_DPU)
        target_compile_definitions(${TARGET} PRIVATE "LOG_DPU")
    endif(LOG_DPU)
    target_compile_definitions(${TARGET} PRIVATE
        DPU_BINARIES_DIR="${DPU_BINARIES_ABS_DIR}"
    )
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${TARGET} PRIVATE ${DPU_LINK})
    target_link_libraries(${TARGET} PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
    target_link_libraries(${TARGET} PRIVATE stdc++fs)
    if(FSANITIZE)
        target_link_options(${TARGET} PRIVATE -fsanitize=leak)
    endif(FSANITIZE)
endfunction()

# ------------------------------------------------------------------------------
#                                Add directories                                
# ------------------------------------------------------------------------------

execute_process(COMMAND dpu-pkg-config --cflags dpu OUTPUT_VARIABLE DPU_H_INCLUDE)
string(REPLACE "-I" "" DPU_H_INCLUDE ${DPU_H_INCLUDE})
string(REPLACE "\n" "" DPU_H_INCLUDE ${DPU_H_INCLUDE})

include_directories(src thirdparty ${DPU_H_INCLUDE})
get_filename_component(DPU_BINARIES_ABS_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/dpu REALPATH)
add_compile_definitions(DPU_BINARIES_DIR="${DPU_BINARIES_ABS_DIR}")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

# ------------------------------------------------------------------------------
#                               Copy some scripts                               
# ------------------------------------------------------------------------------

set(UTILITY_SCRIPTS dtrace.sh)
foreach(SCRIPT ${UTILITY_SCRIPTS})
    configure_file(${SCRIPT} ${SCRIPT} COPYONLY)
endforeach(SCRIPT ${UTILITY_SCRIPTS})

# ------------------------------------------------------------------------------
#                                Final cleaning                                 
# ------------------------------------------------------------------------------

unset(LOG_DPU CACHE)