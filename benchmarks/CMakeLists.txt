set(TARGETS benchmark1)

foreach(TARGET ${TARGETS})

    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} PRIVATE ${LIB_NAME})

endforeach(TARGET ${TARGETS})