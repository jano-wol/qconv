set(target relu-bench)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp
            )
add_executable(${target} ${sources})
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../benchutils/include")
target_link_libraries(${target} PUBLIC benchmark::benchmark layers testutils)

