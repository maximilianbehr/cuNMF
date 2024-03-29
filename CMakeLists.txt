# MIT License
#
# Copyright (c) 2024 Maximilian Behr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.23)
project(cunmf LANGUAGES CXX CUDA VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# check cuda version
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_GREATER_EQUAL "11.4.2")
    message(STATUS "CUDA version ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION} is supported.")
else()
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION} is not supported. Please install CUDA version 11.4.2 or higher.")
endif()

# cunmf library
add_library(cunmf SHARED cunmf.cu cunmf_info.cpp cunmf_options.cpp)
set_property(TARGET cunmf PROPERTY CUDA_ARCHITECTURES all)
set_target_properties(cunmf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cunmf PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(cunmf PUBLIC cublas)
target_compile_options(cunmf PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --threads 0>)
install(TARGETS cunmf DESTINATION lib)
install(FILES cunmf.h DESTINATION include)

# copy binary files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/X_10000_500.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/W0_10000_40.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/H0_40_500.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# examples
add_executable(example_cunmf_MUbeta example_cunmf_MUbeta.cu)
target_link_libraries(example_cunmf_MUbeta PUBLIC cunmf)
set_property(TARGET example_cunmf_MUbeta PROPERTY CUDA_ARCHITECTURES all)
