cmake_minimum_required(VERSION 2.6)
project(transmitchannelizer)

add_executable(transmitchannelizer main.cpp)

install(TARGETS transmitchannelizer RUNTIME DESTINATION bin)
