set(target layer-tests)

set(sources
    TestLinear.cpp
	TestQConv.cpp
)

add_executable(${target} ${sources})

target_link_libraries(${target}
	PUBLIC
	layers
	simd
	testutils
	PRIVATE
	gmock-dev
	GTest::gtest_main
)

gtest_discover_tests(${target})

