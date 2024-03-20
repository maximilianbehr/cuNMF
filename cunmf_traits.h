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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <limits>

template <typename T>
struct cunmf_traits;

template <>
struct cunmf_traits<float> {
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr float one = 1.f;
    static constexpr float zero = 0.f;
    static constexpr float eps = std::numeric_limits<float>::epsilon();

    /*-----------------------------------------------------------------------------
     * maximum value, natural logarithm
     *-----------------------------------------------------------------------------*/
    __device__ inline static float maximum(const float a, const float b) {
        return fmaxf(a, b);
    }

    __device__ inline static float logarithm(const float x) {
        return logf(x);
    }

    __device__ inline static float power(const float x, const float y) {
        return powf(x, y);
    }

    __device__ inline static float squareroot(const float x) {
        return sqrtf(x);
    }

    /*-----------------------------------------------------------------------------
     * matrix mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

template <>
struct cunmf_traits<double> {
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr double one = 1.;
    static constexpr double zero = 0.;
    static constexpr double eps = std::numeric_limits<double>::epsilon();

    /*-----------------------------------------------------------------------------
     * maximum value, natural logarithm
     *-----------------------------------------------------------------------------*/
    __device__ inline static double maximum(const double a, const double b) {
        return max(a, b);
    }

    __device__ inline static double logarithm(const double x) {
        return log(x);
    }

    __device__ inline static double power(const double x, const double y) {
        return pow(x, y);
    }

    __device__ inline static double squareroot(const double x) {
        return sqrt(x);
    }

    /*-----------------------------------------------------------------------------
     * matrix mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};
