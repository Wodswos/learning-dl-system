# 添加可执行文件
add_executable(test_math test_math.cpp)

# 链接 Google Test 库
target_link_libraries(test_math ${GTEST_BOTH_LIBRARIES} pthread)

# 添加测试
add_test(
    NAME test_math
    COMMAND test_math
)