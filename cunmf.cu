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

#if 0
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "checkcuda.h"

template <typename T>
__global__ static void cuexpm_absrowsums(const int n, const T *__restrict__ d_A, const int ldA, double *__restrict__ buffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        double tmp = 0.;
        for (int k = 0; k < n; ++k) {
            tmp += cuexpm_traits<T>::abs(d_A[k + j * ldA]);
        }
        buffer[j] = tmp;
    }
}

template <typename T>
static int cuexpm_matrix1norm(const int n, const T *__restrict__ d_A, const int ldA, void *d_buffer, double *__restrict__ d_nrmA1) {
    *d_nrmA1 = 0.;
    double *buffer = reinterpret_cast<double *>(d_buffer);
    cuexpm_absrowsums<<<(n + 255) / 256, 256>>>(n, d_A, ldA, buffer);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    *d_nrmA1 = *(thrust::max_element(thrust::device_pointer_cast(buffer), thrust::device_pointer_cast(buffer + n)));
    return 0;
}

template <typename T>
static int cuexpm_parameters(const int n, const T *__restrict__ d_A, const int ldA, void *d_buffer, int *m, int *s) {
    double eta1 = 0.;
    CHECK_CUEXPM(cuexpm_matrix1norm(n, d_A, ldA, d_buffer, &eta1));
    if constexpr (std::is_same<T, double>::value || std::is_same<T, cuDoubleComplex>::value) {
        const double theta[] = {1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001, 2.97847961257068e+000, 5.371920351148152e+000};
        *s = 0;
        if (eta1 <= theta[0]) {
            *m = 3;
            return 0;
        }
        if (eta1 <= theta[1]) {
            *m = 5;
            return 0;
        }
        if (eta1 <= theta[2]) {
            *m = 7;
            return 0;
        }
        if (eta1 <= theta[3]) {
            *m = 9;
            return 0;
        }
        *s = ceil(log2(eta1 / theta[4]));
        if (*s < 0) {
            *s = 0;
        }
        *m = 13;
    } else {
        const double theta[] = {4.258730016922831e-001, 1.880152677804762e+000, 3.925724783138660e+000};
        *s = 0;
        if (eta1 <= theta[0]) {
            *m = 3;
            return 0;
        }
        if (eta1 <= theta[1]) {
            *m = 5;
            return 0;
        }
        *s = ceil(log2(eta1 / theta[2]));
        if (*s < 0) {
            *s = 0;
        }
        *m = 7;
    }
    return 0;
}

template <typename T>
__global__ static void setDiag(const int n, T *d_A, const int ldA, const T alpha) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = i0; i < n; i += blockDim.x * gridDim.x) {
        for (int j = j0; j < n; j += blockDim.y * gridDim.y) {
            if (i == j) {
                d_A[i + j * ldA] = alpha;
            } else {
                d_A[i + j * ldA] = cuexpm_traits<T>::zero;
            }
        }
    }
}

__device__ static cuComplex &operator+=(cuComplex &a, const cuComplex &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ static cuDoubleComplex &operator+=(cuDoubleComplex &a, const cuDoubleComplex &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <typename T>
__global__ static void addDiag(const int n, T *d_A, const int ldA, const T alpha) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        d_A[j + j * ldA] += alpha;
    }
}

const static cusolverAlgMode_t CUSOLVER_ALG = CUSOLVER_ALG_0;

template <typename T>
static int cuexpm_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    /*-----------------------------------------------------------------------------
     * initialize with zero
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = 0;
    *h_bufferSize = 0;

    /*-----------------------------------------------------------------------------
     * get device and host workspace size for LU factorization
     *-----------------------------------------------------------------------------*/
    // create cusolver handle
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // create cusolver params
    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    // compute workspace size
    CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cuexpm_traits<T>::dataType, nullptr, n, cuexpm_traits<T>::computeType, d_bufferSize, h_bufferSize));

    // free workspace
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));

    /*-----------------------------------------------------------------------------
     * compute final workspace size
     * matrix T1, T2, T4, T6, T8, U, V -> n * n * 5 +  n * n * 2
     * int64 array ipiv -> n * sizeof(int64_t)
     * int info -> sizeof(int)
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = std::max(*d_bufferSize, sizeof(T) * n * n * 5) + sizeof(T) * n * n * 2 + sizeof(int64_t) * n + sizeof(int);

    return 0;
}

int cuexpmd_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<double>(n, d_bufferSize, h_bufferSize);
}

int cuexpms_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<float>(n, d_bufferSize, h_bufferSize);
}

int cuexpmc_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<cuComplex>(n, d_bufferSize, h_bufferSize);
}

int cuexpmz_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<cuDoubleComplex>(n, d_bufferSize, h_bufferSize);
}

template <typename T>
static int cuexpm(const int n, const T *d_A, const int ldA, void *d_buffer, void *h_buffer, T *d_F, const int ldF) {
    /*-----------------------------------------------------------------------------
     * kernel launch parameters
     *-----------------------------------------------------------------------------*/
    const size_t threadsPerBlock = 256;                                        // addDiag
    const size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // addDiag
    dim3 grid((n + 15) / 16, (n + 15) / 16);                                   // setDiag
    dim3 block(16, 16);                                                        // setDiag

    /*-----------------------------------------------------------------------------
     * compute the scaling parameter and Pade approximant degree
     *-----------------------------------------------------------------------------*/
    int m, s;
    CHECK_CUEXPM(cuexpm_parameters(n, d_A, ldA, d_buffer, &m, &s));
    // printf("m = %d, s = %d\n", m, s);

    /*-----------------------------------------------------------------------------
     * split memory buffer
     * memory layout: |U, V, T1, T2, T4, T6, T8| from 0 to (n * n * 7) -1
     *-----------------------------------------------------------------------------*/
    T *T1, *T2, *T4, *T6, *T8, *U, *V;
    U = (T *)d_buffer;
    V = U + n * n * 1;
    T1 = U + n * n * 2;
    T2 = U + n * n * 3;
    T4 = U + n * n * 4;
    T6 = U + n * n * 5;
    T8 = U + n * n * 6;

    /*-----------------------------------------------------------------------------
     * create cuBlas handle
     *-----------------------------------------------------------------------------*/
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * rescale T, T = T / 2^s
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, d_A, ldA, &cuexpm_traits<T>::zero, T1, n, T1, n));

    typename cuexpm_traits<T>::S alpha = 1. / (1 << s);
    if (s != 0) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXdscal(cublasH, n * n, &alpha, T1, 1));
    }

    /*-----------------------------------------------------------------------------
     * compute powers of T if needed
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T1, n, T1, n, &cuexpm_traits<T>::zero, T2, n));
    if (m >= 5) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T2, n, T2, n, &cuexpm_traits<T>::zero, T4, n));
    }
    if (m >= 7) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T2, n, T4, n, &cuexpm_traits<T>::zero, T6, n));
    }
    if (m == 9) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T4, n, T4, n, &cuexpm_traits<T>::zero, T8, n));
    }

    /*-----------------------------------------------------------------------------
     * compute U and V for the Pade approximant independently on different streams
     *-----------------------------------------------------------------------------*/
    cudaStream_t streamU, streamV;
    CHECK_CUDA(cudaStreamCreate(&streamU));
    CHECK_CUDA(cudaStreamCreate(&streamV));
    if (m == 3) {
        // U = U + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(n, U, n, cuexpm_traits<T>::Pade3[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade3[3], T2, n, U, n));

        // V = V + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(n, V, n, cuexpm_traits<T>::Pade3[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade3[2], T2, n, V, n));
    } else if (m == 5) {
        // U = U + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(n, U, n, cuexpm_traits<T>::Pade5[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade5[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade5[5], T4, n, U, n));

        // V = V + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(n, V, n, cuexpm_traits<T>::Pade5[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade5[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade5[4], T4, n, V, n));
    } else if (m == 7) {
        // U = U + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(n, U, n, cuexpm_traits<T>::Pade7[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade7[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade7[5], T4, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade7[7], T6, n, U, n));

        // V = V + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(n, V, n, cuexpm_traits<T>::Pade7[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade7[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade7[4], T4, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade7[6], T6, n, V, n));
    } else if (m == 9) {
        // U = U + c(9)*T8 + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(n, U, n, cuexpm_traits<T>::Pade9[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[5], T4, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[7], T6, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[9], T8, n, U, n));

        // V = V + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(n, V, n, cuexpm_traits<T>::Pade9[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[4], T4, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[6], T6, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[8], T8, n, V, n));
    } else if (m == 13) {
        //  U = T6*(c(13)*T6 + c(11)*T4 + c(9)*T2) + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I;
        setDiag<<<grid, block, 0, streamU>>>(n, U, n, cuexpm_traits<T>::Pade13[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade13[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade13[5], T4, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade13[7], T6, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[9], T2, n, &cuexpm_traits<T>::Pade13[11], T4, n, T8, n));  // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[13], T6, n, &cuexpm_traits<T>::one, T8, n, T8, n));        // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T6, n, T8, n, &cuexpm_traits<T>::one, U, n));

        // V = T6*(c(12)*T6 + c(10)*T4 + c(8)*T2) + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I;
        setDiag<<<grid, block, 0, streamV>>>(n, V, n, cuexpm_traits<T>::Pade13[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade13[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade13[4], T4, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade13[6], T6, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[8], T2, n, &cuexpm_traits<T>::Pade13[10], T4, n, T8, n));  // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[12], T6, n, &cuexpm_traits<T>::one, T8, n, T8, n));        // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T6, n, T8, n, &cuexpm_traits<T>::one, V, n));
    } else {
        fprintf(stderr, "m must be 3, 5, 7, 9, or 13\n");
        fflush(stderr);
        return -1;
    }
    CHECK_CUDA(cudaStreamSynchronize(streamU));
    CHECK_CUDA(cudaStreamSynchronize(streamV));
    CHECK_CUDA(cudaStreamDestroy(streamU));
    CHECK_CUDA(cudaStreamDestroy(streamV));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    // U = T*U
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T1, n, U, n, &cuexpm_traits<T>::zero, T8, n));
    std::swap(U, T8);

    /*-----------------------------------------------------------------------------
     *  compute F = (V-U)\(U+V) = (V-U)\2*U + I
     *-----------------------------------------------------------------------------*/
    // prepare right-hand side
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::mone, U, n, &cuexpm_traits<T>::one, V, n, V, n));

    typename cuexpm_traits<T>::S two = 2.;
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXdscal(cublasH, n * n, &two, U, 1));

    // create cusolver handle
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // create cusolver params
    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    // split the memory buffer
    // memory layout: |U, V, T1, T2, T4, T6, T8| from 0 to (n * n * 7) -1
    //                |x, x, ipiv, info, dwork|  from n * n * 2         to n * n * 2 + n -1 (ipiv) overwrites T1
    //                                           from n * n * 2 + n     to n * n * 2 + n    (info) overwrites T1, T2
    //                                           from n * n * 2 + n + 1 to XXXXXXXXXXXXX    (dwork) overwrites T1, T2, T4, T6, T8, ...
    int64_t *d_ipiv = (int64_t *)T1;    // use T1 as ipiv
    int *d_info = (int *)(d_ipiv + n);  // put d_info after d_ipiv
    void *d_work = d_info + 1;          // put d_work after d_info
    void *h_work = h_buffer;            // use h_buffer as workspace

    // compute LU factorization
    size_t lworkdevice = 0, lworkhost = 0;
    CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cuexpm_traits<T>::dataType, V, n, cuexpm_traits<T>::computeType, &lworkdevice, &lworkhost));
    CHECK_CUSOLVER(cusolverDnXgetrf(cusolverH, params, n, n, cuexpm_traits<T>::dataType, V, n, d_ipiv, cuexpm_traits<T>::computeType, d_work, lworkdevice, h_work, lworkhost, d_info));

    // solve linear system
    CHECK_CUSOLVER(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n, cuexpm_traits<T>::dataType, V, n, d_ipiv, cuexpm_traits<T>::computeType, U, n, d_info));

    // free workspace
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));

    // add identity
    addDiag<<<blocksPerGrid, threadsPerBlock>>>(n, U, n, cuexpm_traits<T>::one);
    CHECK_CUDA(cudaPeekAtLastError());

    /*-----------------------------------------------------------------------------
     * squaring phase
     *-----------------------------------------------------------------------------*/
    if (((s % 2) == 0) && s > 0) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, U, n, U, n, &cuexpm_traits<T>::zero, V, n));
        std::swap(U, V);
        s--;
    }

    int ldU = n;
    int ldF_ = ldF;  // save original ldF
    for (int k = 0; k < s; ++k) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, U, ldU, U, ldU, &cuexpm_traits<T>::zero, d_F, ldF_));
        std::swap(U, d_F);
        std::swap(ldU, ldF_);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    /*-----------------------------------------------------------------------------
     * free memory and destroy cuBlas handle
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cublasDestroy(cublasH));
    return 0;
}

int cuexpms(const int n, const float *d_A, const int ldA, void *d_buffer, void *h_buffer, float *d_expmA, const int ldexpmA) {
    return cuexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

int cuexpmd(const int n, const double *d_A, const int ldA, void *d_buffer, void *h_buffer, double *d_expmA, const int ldexpmA) {
    return cuexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

int cuexpmc(const int n, const cuComplex *d_A, const int ldA, void *d_buffer, void *h_buffer, cuComplex *d_expmA, const int ldexpmA) {
    return cuexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

int cuexpmz(const int n, const cuDoubleComplex *d_A, const int ldA, void *d_buffer, void *h_buffer, cuDoubleComplex *d_expmA, const int ldexpmA) {
    return cuexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

#endif
