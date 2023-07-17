find_package(OpenMP REQUIRED)
find_package(Catch2 3 REQUIRED)

set(DPU_BINARIES_DIR "dpu")
get_filename_component(DPU_BINARIES_ABS_DIR ${CMAKE_CURRENT_BINARY_DIR}/${DPU_BINARIES_DIR} REALPATH)
add_subdirectory(${DPU_BINARIES_DIR})

execute_process(COMMAND dpu-pkg-config --cflags dpu OUTPUT_VARIABLE DPU_H_INCLUDE)
string(REPLACE "-I" "" DPU_H_INCLUDE ${DPU_H_INCLUDE})
string(REPLACE "\n" "" DPU_H_INCLUDE ${DPU_H_INCLUDE})

execute_process(COMMAND dpu-pkg-config --libs dpu OUTPUT_VARIABLE DPU_LINK)
string(REPLACE "\n" "" DPU_LINK ${DPU_LINK})

set(TARGETS unit_test1 benchmark1)

foreach(TARGET ${TARGETS})

    add_executable(${TARGET} ${TARGET}.cpp)

    target_include_directories(${TARGET} PUBLIC "." ${DPU_H_INCLUDE})
    if (LOG_DPU)
        target_compile_definitions(${TARGET} PRIVATE "LOG_DPU")
    endif()
    target_compile_definitions(${TARGET} PRIVATE
        DPU_BINARIES_DIR="${DPU_BINARIES_ABS_DIR}"
    )
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${TARGET} PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(${TARGET} PRIVATE ${DPU_LINK})

endforeach()