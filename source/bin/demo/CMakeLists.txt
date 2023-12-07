set(target demo)
set(sources source/main.cpp)
add_executable(${target} ${sources})
target_link_libraries(${target} PUBLIC layers)
