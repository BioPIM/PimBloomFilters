# ------------------------------------------------------------------------------
#                                 Dependencies                                  
# ------------------------------------------------------------------------------

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)

# ------------------------------------------------------------------------------
#                      Compile DPU binaries                      
# ------------------------------------------------------------------------------

add_subdirectory(dpu)

# ------------------------------------------------------------------------------
#                            Retrieve some DPU info                             
# ------------------------------------------------------------------------------

execute_process(COMMAND dpu-pkg-config --libs dpu OUTPUT_VARIABLE DPU_LINK)
string(REPLACE "\n" "" DPU_LINK ${DPU_LINK})

# ------------------------------------------------------------------------------
#                                Compile library                                
# ------------------------------------------------------------------------------

add_library(${LIB_NAME} STATIC
    pim_rankset.cpp
    pim_bloom_filter.cpp
    standard_bloom_filter.cpp
    bloom_filter_factory.cpp
)

target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra -g)
if(FSANITIZE)
    target_compile_options(${LIB_NAME} PRIVATE -fsanitize=leak)
endif(FSANITIZE)
target_link_libraries(${LIB_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${LIB_NAME} PRIVATE ${DPU_LINK})
target_link_libraries(${LIB_NAME} PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
target_link_libraries(${LIB_NAME} PRIVATE stdc++fs)
if(FSANITIZE)
    target_link_options(${LIB_NAME} PRIVATE -fsanitize=leak)
endif(FSANITIZE)