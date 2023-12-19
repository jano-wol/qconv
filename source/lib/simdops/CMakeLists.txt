set(target simdops)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(headers ${include_path}/simdops.h
            )
set(headers ${source_path}/simdops.cpp
            )            
add_library(${target} ${headers} ${sources})
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(${target} PUBLIC simde)
