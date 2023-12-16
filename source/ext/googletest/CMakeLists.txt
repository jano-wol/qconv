set(gmock_build_tests             OFF CACHE BOOL "")
set(gtest_build_samples           OFF CACHE BOOL "")
set(gtest_build_tests             OFF CACHE BOOL "")
set(gtest_disable_pthreads        OFF CACHE BOOL "")
set(gtest_force_shared_crt        ON  CACHE BOOL "")
set(gtest_hide_internal_symbols   OFF CACHE BOOL "")

add_subdirectory(googletest-release-1.12.1)

add_library(gmock-dev INTERFACE)

target_include_directories(gmock-dev
		SYSTEM INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/googletest-release-1.12.1/googletest/include
		${CMAKE_CURRENT_SOURCE_DIR}/googletest-release-1.12.1/googlemock/include
		)
		
target_link_libraries(gmock-dev
		INTERFACE
		gmock
)		
