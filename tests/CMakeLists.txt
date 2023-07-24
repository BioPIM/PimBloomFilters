find_package(Catch2 3 REQUIRED)

set(TARGETS unit_test1 unit_test2)

foreach(TARGET ${TARGETS})

add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} PRIVATE ${LIB_NAME})
    target_link_libraries(${TARGET} PRIVATE Catch2::Catch2WithMain)
endforeach(TARGET ${TARGETS})