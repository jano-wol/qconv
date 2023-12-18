set(target simdops)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(headers ${include_path}/simdops.h
            )
add_library(${target} ${headers})
set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(${target} PUBLIC simde)
