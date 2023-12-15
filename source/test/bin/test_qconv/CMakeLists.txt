set(target test_qconv)
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(sources ${source_path}/main.cpp)
add_executable(${target} ${sources})
target_link_libraries(${target} PUBLIC core layers)
set_target_properties(${target}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/bin"
)
