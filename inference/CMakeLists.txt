cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(LLMINFER)

# 查找 Google Test 库
find_package(GTest REQUIRED)

# 添加包含目录
include_directories(${GTEST_INCLUDE_DIRS})

# 添加测试目录
add_subdirectory(test)