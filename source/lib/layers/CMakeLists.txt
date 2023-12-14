set(target layers)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(headers ${include_path}/add.h
            ${include_path}/bias.h
			${include_path}/common.h
            ${include_path}/qconv.h
			${include_path}/tile.h
            )
set(sources ${source_path}/tile.cpp
            )			
add_library(${target} ${headers} ${sources})
set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(${target} core)
