set(TARGETS bloom_filters_dpu)

set(TASKLETS "16")
set(STACK_SIZE "3072")

foreach(TARGET ${TARGETS})

    add_executable(${TARGET} ${TARGET}.c)

    target_compile_options(${TARGET} PRIVATE --target=dpu-upmem-dpurte)
    target_link_options(${TARGET} PRIVATE --target=dpu-upmem-dpurte)
    target_compile_definitions(${TARGET} PRIVATE
        NR_TASKLETS=${TASKLETS}
        STACK_SIZE_DEFAULT=${STACK_SIZE}
    )
    if (LOG_DPU)
        target_compile_definitions(${TARGET} PRIVATE "LOG_DPU")
    endif(LOG_DPU)
    target_link_options(${TARGET} PRIVATE
        -DNR_TASKLETS=${TASKLETS}
        -DSTACK_SIZE_DEFAULT=${STACK_SIZE}
    )

endforeach(TARGET ${TARGETS})