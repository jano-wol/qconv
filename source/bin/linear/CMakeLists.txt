set(target linear-bin)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp
            )
add_executable(${target} ${sources})
target_link_libraries(${target} PUBLIC layers testutils)

