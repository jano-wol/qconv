set(target operations-tests)

set(sources
	TestMinMaxH.cpp
)

add_executable(${target} ${sources})

target_link_libraries(${target}
	PUBLIC
	simd
	testutils
	PRIVATE
	gmock-dev
	GTest::gtest_main
)

gtest_discover_tests(${target})

