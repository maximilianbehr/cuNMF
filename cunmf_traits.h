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

template <typename T>
struct cunmf_traits;

template <>
struct cunmf_traits<float> {
    typedef float S;
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr float one = 1.f;
    static constexpr float mone = -1.f;
    static constexpr float zero = 0.f;

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const double x) {
        return fabsf(x);
    }

    /*-----------------------------------------------------------------------------
     * matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

template <>
struct cunmf_traits<double> {
    typedef double S;
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr double one = 1.;
    static constexpr double mone = -1.;
    static constexpr double zero = 0.;

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const double x) {
        return fabs(x);
    }

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};
