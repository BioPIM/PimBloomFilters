find_package(OpenMP REQUIRED)

set(TARGETS benchmark1 benchmark2)

foreach(TARGET ${TARGETS})

    add_executable(${TARGET} ${TARGET}.cpp)
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -g)
    target_link_libraries(${TARGET} PRIVATE ${LIB_NAME})
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)

endforeach(TARGET ${TARGETS})