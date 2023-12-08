set(target core)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCORELIBFOLDERPATH=${CMAKE_CURRENT_SOURCE_DIR}")
set(headers ${include_path}/paths.h
            ${include_path}/simdops.h
            )
set(sources ${source_path}/paths.cpp
			)
add_library(${target} ${sources} ${headers})
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(${target} PUBLIC simde)
