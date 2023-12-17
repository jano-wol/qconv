set(target layer-tests)

set(sources
	TestQConv.cpp
)

add_executable(${target} ${sources})

target_link_libraries(${target}
	PUBLIC
	core
	layers
	testutils
	PRIVATE
	gmock-dev
	GTest::gtest_main
)

gtest_discover_tests(${target})

