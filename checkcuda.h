/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cublas_v2.h>

#include <cstdio>

#include "cunmf_error.h"

#define CHECK_CUNMF(err)                                                                \
    do {                                                                                \
        int error_code = (err);                                                         \
        if (error_code) {                                                               \
            fprintf(stderr, "cunmf error %d. %s:%d\n", error_code, __FILE__, __LINE__); \
            fflush(stderr);                                                             \
            return CUNMF_INTERNAL_ERROR;                                                \
        }                                                                               \
    } while (false)

#define CHECK_CUDA(err)                                                                                                    \
    do {                                                                                                                   \
        cudaError_t error_code = (err);                                                                                    \
        if (error_code != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA Error %d: %s. %s:%d\n", error_code, cudaGetErrorString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                                                                \
            return CUNMF_CUDA_ERROR;                                                                                       \
        }                                                                                                                  \
    } while (false)

#define CHECK_CUBLAS(err)                                                                                                        \
    do {                                                                                                                         \
        cublasStatus_t error_code = (err);                                                                                       \
        if (error_code != CUBLAS_STATUS_SUCCESS) {                                                                               \
            fprintf(stderr, "CUBLAS Error %d - %s. %s:%d\n", error_code, cublasGetStatusString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                                                                      \
            return CUNMF_CUBLAS_ERROR;                                                                                           \
        }                                                                                                                        \
    } while (false)
