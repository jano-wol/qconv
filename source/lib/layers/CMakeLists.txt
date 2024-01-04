set(target layers)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(headers ${include_path}/linear.h
            ${include_path}/linear_naive.h
            ${include_path}/qconv.h
            ${include_path}/qconv_naive.h
            )
set(sources ${source_path}/linear.cpp
            ${source_path}/linear_naive.cpp
            ${source_path}/qconv.cpp
            ${source_path}/qconv_naive.cpp
            )			
add_library(${target} ${headers} ${sources})
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(${target} simd)
