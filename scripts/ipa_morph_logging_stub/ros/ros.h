#pragma once
// Standalone build only: Morph uses ROS solely for these progress messages.
// No algorithm, data, or parameter behavior is substituted here.
#include <cstdio>
#define ROS_INFO(...) do { std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)
