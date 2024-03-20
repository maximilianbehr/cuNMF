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

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include <chrono>

#include "checkcuda.h"
#include "cunmf.h"
#include "cunmf_traits.h"

template <typename T>
struct square : public thrust::unary_function<T, double> {
    __host__ __device__ double operator()(const T &x) const {
        return x * x;
    }
};

template <typename T>
static int cunmf_normFro(int m, int n, const T *A, double *nrmA) {
    /*-----------------------------------------------------------------------------
     * compute || A ||_F = sqrt( sum( A(:).^2 ) )
     *-----------------------------------------------------------------------------*/
    *nrmA = thrust::transform_reduce(thrust::device_pointer_cast(A), thrust::device_pointer_cast(A + m * n), square<T>(), 0.0, thrust::plus<double>());
    *nrmA = sqrt(*nrmA);
    return 0;
}

template <typename T>
struct diffsquare : public thrust::binary_function<T, T, double> {
    __host__ __device__ double operator()(const T &x, const T &y) const {
        T z = x - y;
        return z * z;
    }
};

template <typename T>
static int cunmf_diffnormFro(int m, int n, const T *A, const T *B, double *diff) {
    /*-----------------------------------------------------------------------------
     * compute || A - B ||_F = sqrt( sum( (A(:) - B(:)).^2 ) )
     *-----------------------------------------------------------------------------*/
    *diff = thrust::inner_product(thrust::device_pointer_cast(A), thrust::device_pointer_cast(A + m * n), thrust::device_pointer_cast(B), 0.0, thrust::plus<double>(), diffsquare<T>());
    *diff = sqrt(*diff);
    return 0;
}

template <typename T>
static int cunmf_objFro(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *W, const T *H,
                        void *buffer, double *objFro) {
    /*-----------------------------------------------------------------------------
     * compute W*H
     *-----------------------------------------------------------------------------*/
    T *WH = reinterpret_cast<T *>(buffer);
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, H, k, &cunmf_traits<T>::zero, WH, m));

    /*-----------------------------------------------------------------------------
     * compute 0.5 * || X - WH ||_F^2
     *-----------------------------------------------------------------------------*/
    CHECK_CUNMF(cunmf_diffnormFro(m, n, X, WH, objFro));
    *objFro = 0.5 * (*objFro) * (*objFro);
    return 0;
}

template <typename T>
__global__ void MUupdate(int m, int k, const T *numerator, const T *denominator, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = Wp .* (numerator ./ (denominator + eps))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = Wp[i + j * m] * numerator[i + j * m] / (denominator[i + j * m] + cunmf_traits<T>::eps);
        }
    }
}

template <typename T>
__global__ void MUupdate(int m, int k, const T gamma, const T *numerator, const T *denominator, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = Wp .* (numerator ./ (denominator + eps)).^gamma
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = Wp[i + j * m] * cunmf_traits<T>::power(numerator[i + j * m] / (denominator[i + j * m] + cunmf_traits<T>::eps), gamma);
        }
    }
}

template <typename T>
__global__ void MUupdate(const T eps, int m, int k, const T *numerator, const T *denominator, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = max(eps, Wp .* (numerator ./ (denominator + eps)))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = cunmf_traits<T>::maximum(eps, Wp[i + j * m] * numerator[i + j * m] / (denominator[i + j * m] + cunmf_traits<T>::eps));
        }
    }
}

template <typename T>
__global__ void MUupdate(const T eps, int m, int k, const T gamma, const T *numerator, const T *denominator, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = max(eps, Wp .* (numerator ./ (denominator + eps)).^gamma)
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = cunmf_traits<T>::maximum(eps, Wp[i + j * m] * cunmf_traits<T>::power(numerator[i + j * m] / (denominator[i + j * m] + cunmf_traits<T>::eps), gamma));
        }
    }
}

template <typename T>
__global__ void MUsqrtupdate(int m, int k, const T *numerator, const T *denominator, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = Wp .* sqrt(numerator ./ (denominator + eps))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = Wp[i + j * m] * cunmf_traits<T>::squareroot(numerator[i + j * m] / (denominator[i + j * m] + cunmf_traits<T>::eps));
        }
    }
}

template <typename T>
__global__ void MUsqrtupdate(const T eps, int m, int k, const T *numerator, const T *denominator, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = max(eps, Wp .* sqrt(numerator ./ (denominator + eps)))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = cunmf_traits<T>::maximum(eps, Wp[i + j * m] * cunmf_traits<T>::squareroot(numerator[i + j * m] / (denominator[i + j * m] + cunmf_traits<T>::eps)));
        }
    }
}

template <typename T>
static int updateW_fro(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *Wp, const T *Hp, void *buffer, cunmf_options opt, T *W) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for XHpT, HpHpT, WpHpHpT
     *-----------------------------------------------------------------------------*/
    T *XHpT = reinterpret_cast<T *>(buffer);
    T *HpHpT = reinterpret_cast<T *>(XHpT + m * k);
    T *WpHpHpT = reinterpret_cast<T *>(HpHpT + k * k);

    /*-----------------------------------------------------------------------------
     * compute numerator: X*H^T m-x-k on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream1));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &cunmf_traits<T>::one, X, m, Hp, k, &cunmf_traits<T>::zero, XHpT, m));

    /*-----------------------------------------------------------------------------
     * compute denumerator: H*H^T k-x-k on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream2));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, k, k, n, &cunmf_traits<T>::one, Hp, k, Hp, k, &cunmf_traits<T>::zero, HpHpT, k));

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    /*-----------------------------------------------------------------------------
     * compute denumerator: W*H*H^T m-x-k
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, k, k, &cunmf_traits<T>::one, Wp, m, HpHpT, k, &cunmf_traits<T>::zero, WpHpHpT, m));

    /*-----------------------------------------------------------------------------
     * update W
     *-----------------------------------------------------------------------------*/
    dim3 grid((m + 15) / 16, (k + 15) / 16);
    dim3 block(16, 16);
    if (opt->eps == 0.0) {
        MUupdate<<<grid, block>>>(m, k, XHpT, WpHpHpT, Wp, W);
    } else {
        MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), m, k, XHpT, WpHpHpT, Wp, W);
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
static int updateH_fro(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *W, const T *Hp, void *buffer, cunmf_options opt, T *H) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for WTX, WTW, WTWHp
     *-----------------------------------------------------------------------------*/
    T *WTX = reinterpret_cast<T *>(buffer);
    T *WTW = reinterpret_cast<T *>(WTX + k * n);
    T *WTWHp = reinterpret_cast<T *>(WTW + k * k);

    /*-----------------------------------------------------------------------------
     * compute numerator: W^T*X k-x-n on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream1));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &cunmf_traits<T>::one, W, m, X, m, &cunmf_traits<T>::zero, WTX, k));

    /*-----------------------------------------------------------------------------
     * compute denumerator: W^T*W k-x-k on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream2));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, k, m, &cunmf_traits<T>::one, W, m, W, m, &cunmf_traits<T>::zero, WTW, k));

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    /*-----------------------------------------------------------------------------
     * compute denumerator: W^T*W*H k-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, k, n, k, &cunmf_traits<T>::one, WTW, k, Hp, k, &cunmf_traits<T>::zero, WTWHp, k));

    /*-----------------------------------------------------------------------------
     * update H
     *-----------------------------------------------------------------------------*/
    dim3 grid((k + 15) / 16, (n + 15) / 16);
    dim3 block(16, 16);
    if (opt->eps == 0.0) {
        MUupdate<<<grid, block>>>(k, n, WTX, WTWHp, Hp, H);
    } else {
        MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), k, n, WTX, WTWHp, Hp, H);
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
struct ISdiv : public thrust::binary_function<T, T, double> {
    __host__ __device__ double operator()(const T &x, const T &y) const {
        T frac = x / (y + cunmf_traits<T>::eps);
        return frac - cunmf_traits<T>::logarithm(frac + cunmf_traits<T>::eps) - 1;
    }
};

template <typename T>
static int cunmf_objIS(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *W, const T *H,
                       void *buffer, double *obj) {
    /*-----------------------------------------------------------------------------
     * compute W*H
     *-----------------------------------------------------------------------------*/
    T *WH = reinterpret_cast<T *>(buffer);
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, H, k, &cunmf_traits<T>::zero, WH, m));

    /*-----------------------------------------------------------------------------
     * compute IS divergence
     *-----------------------------------------------------------------------------*/
    *obj = thrust::inner_product(thrust::device_pointer_cast(X), thrust::device_pointer_cast(X + m * n), thrust::device_pointer_cast(WH), 0.0, thrust::plus<double>(), ISdiv<T>());

    return 0;
}

template <typename T>
__global__ void prepare_update_IS(int m, int n, const T *X, T *WHm1, T *WHm2X) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute (W*H).^(-1) and (W*H).^(-2).*X
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < n; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            T tmp = WHm1[i + j * m];
            WHm1[i + j * m] = cunmf_traits<T>::one / (tmp + cunmf_traits<T>::eps);
            WHm2X[i + j * m] = cunmf_traits<T>::one / (tmp * tmp + cunmf_traits<T>::eps) * X[i + j * m];
        }
    }
}

template <typename T>
static int updateW_IS(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *Wp, const T *Hp, void *buffer, cunmf_options opt, T *W) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for
     *-----------------------------------------------------------------------------*/
    T *WHm1 = reinterpret_cast<T *>(buffer);             // (W*H).^(-1)
    T *WHm2X = reinterpret_cast<T *>(WHm1 + m * n);      // (W*H).^(-2).*X
    T *WHm1HT = reinterpret_cast<T *>(WHm2X + m * n);    // (W*H).^(-1)*H^T
    T *WHm2XHT = reinterpret_cast<T *>(WHm1HT + m * k);  // (W*H).^(-2).*X*H^T

    /*-----------------------------------------------------------------------------
     * compute: W*H m-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, Wp, m, Hp, k, &cunmf_traits<T>::zero, WHm1, m));

    /*-----------------------------------------------------------------------------
     * compute: (W*H).^(-1) and (W*H).^(-2).*X m-x-n
     *-----------------------------------------------------------------------------*/
    {
        dim3 grid((m + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        prepare_update_IS<<<grid, block>>>(m, n, X, WHm1, WHm2X);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * compute numerator: (W*H).^(-2).*X*H^T m-x-k on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream1));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &cunmf_traits<T>::one, WHm2X, m, Hp, k, &cunmf_traits<T>::zero, WHm2XHT, m));

    /*-----------------------------------------------------------------------------
     * compute denumerator: (W*H).^(-1)*H^T m-x-k on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream2));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &cunmf_traits<T>::one, WHm1, m, Hp, k, &cunmf_traits<T>::zero, WHm1HT, m));

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    /*-----------------------------------------------------------------------------
     * update W
     *-----------------------------------------------------------------------------*/
    dim3 grid((m + 15) / 16, (k + 15) / 16);
    dim3 block(16, 16);
    if (opt->eps == 0.0) {
        MUsqrtupdate<<<grid, block>>>(m, k, WHm2XHT, WHm1HT, Wp, W);
    } else {
        MUsqrtupdate<<<grid, block>>>(static_cast<T>(opt->eps), m, k, WHm2XHT, WHm1HT, Wp, W);
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
static int updateH_IS(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *W, const T *Hp, void *buffer, cunmf_options opt, T *H) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for
     *-----------------------------------------------------------------------------*/
    T *WHm1 = reinterpret_cast<T *>(buffer);             // (W*H).^(-1)
    T *WHm2X = reinterpret_cast<T *>(WHm1 + m * n);      // (W*H).^(-2).*X
    T *WTWHm1 = reinterpret_cast<T *>(WHm2X + m * n);    // W^T*(W*H).^(-1)
    T *WTWHm2X = reinterpret_cast<T *>(WTWHm1 + k * n);  // W^T*(W*H).^(-2).*X

    /*-----------------------------------------------------------------------------
     * compute: W*H m-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, Hp, k, &cunmf_traits<T>::zero, WHm1, m));

    /*-----------------------------------------------------------------------------
     * compute: (W*H)^(-1) and (W*H)^(-2).*X
     *-----------------------------------------------------------------------------*/
    {
        dim3 grid((m + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        prepare_update_IS<<<grid, block>>>(m, n, X, WHm1, WHm2X);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * compute numerator: W^T*(W*H).^(-2).*X k-x-n on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream1));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &cunmf_traits<T>::one, W, m, WHm2X, m, &cunmf_traits<T>::zero, WTWHm2X, k));

    /*-----------------------------------------------------------------------------
     * compute denumerator: W^T*(W*H).^(-1) k-x-n on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream2));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &cunmf_traits<T>::one, W, m, WHm1, m, &cunmf_traits<T>::zero, WTWHm1, k));

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    /*-----------------------------------------------------------------------------
     * update W
     *-----------------------------------------------------------------------------*/
    dim3 grid((k + 15) / 16, (n + 15) / 16);
    dim3 block(16, 16);
    if (opt->eps == 0.0) {
        MUsqrtupdate<<<grid, block>>>(k, n, WTWHm2X, WTWHm1, Hp, H);
    } else {
        MUsqrtupdate<<<grid, block>>>(static_cast<T>(opt->eps), k, n, WTWHm2X, WTWHm1, Hp, H);
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
struct KLdiv : public thrust::binary_function<T, T, double> {
    __host__ __device__ double operator()(const T &x, const T &y) const {
        T frac = x / (y + cunmf_traits<T>::eps);
        return x * cunmf_traits<T>::logarithm(frac + cunmf_traits<T>::eps) - x + y;
    }
};

template <typename T>
static int cunmf_objKL(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *W, const T *H,
                       void *buffer, double *obj) {
    /*-----------------------------------------------------------------------------
     * compute W*H
     *-----------------------------------------------------------------------------*/
    T *WH = reinterpret_cast<T *>(buffer);
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, H, k, &cunmf_traits<T>::zero, WH, m));

    /*-----------------------------------------------------------------------------
     * compute KL divergence
     *-----------------------------------------------------------------------------*/
    *obj = thrust::inner_product(thrust::device_pointer_cast(X), thrust::device_pointer_cast(X + m * n), thrust::device_pointer_cast(WH), 0.0, thrust::plus<double>(), KLdiv<T>());

    return 0;
}

template <typename T>
__global__ void prepare_update_KL(int m, int n, const T *X, T *WHm1X) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute (W*H).^(-1).*X
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < n; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            WHm1X[i + j * m] = X[i + j * m] * cunmf_traits<T>::one / (WHm1X[i + j * m] + cunmf_traits<T>::eps);
        }
    }
}

template <typename T>
__global__ void rowSum(int k, int n, const T *H, T *rowSumsH) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int stridei = blockDim.x * gridDim.x;
    for (int i = i0; i < k; i += stridei) {
        T sum = 0;
        for (int j = 0; j < n; j++) {
            sum += H[i + j * k];
        }
        rowSumsH[i] = sum;
    }
}

template <typename T>
__global__ void MUWupdateKL(int m, int k, const T *numerator, const T *rowSums, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = Wp .* (numerator ./ (rowsums + eps))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = Wp[i + j * m] * numerator[i + j * m] / (rowSums[j] + cunmf_traits<T>::eps);
        }
    }
}

template <typename T>
__global__ void MUWupdateKL(const T eps, int m, int k, const T *numerator, const T *rowSums, const T *Wp, T *W) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute W = max(eps, Wp .* (numerator ./ (rowsums + eps)))
     *-----------------------------------------------------------------------------*/

    for (int j = j0; j < k; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            W[i + j * m] = cunmf_traits<T>::maximum(eps, Wp[i + j * m] * numerator[i + j * m] / (rowSums[j] + cunmf_traits<T>::eps));
        }
    }
}

template <typename T>
static int updateW_KL(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *Wp, const T *Hp, void *buffer, cunmf_options opt, T *W) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for
     *-----------------------------------------------------------------------------*/
    T *WHm1X = reinterpret_cast<T *>(buffer);              // (W*H).^(-1).*X
    T *WHm1XHT = reinterpret_cast<T *>(WHm1X + m * n);     // ((W*H).^(-1).*X)*H^T
    T *rowSumsH = reinterpret_cast<T *>(WHm1XHT + m * k);  // sum(H, 2)

    /*-----------------------------------------------------------------------------
     * compute: W*H m-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, Wp, m, Hp, k, &cunmf_traits<T>::zero, WHm1X, m));

    /*-----------------------------------------------------------------------------
     * compute: (W*H).^(-1).*X on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    {
        dim3 grid((m + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        prepare_update_KL<<<grid, block, 0, stream1>>>(m, n, X, WHm1X);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * compute row sums of H on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    {
        dim3 grid((k + 31) / 32);
        dim3 block(32);
        rowSum<<<grid, block, 0, stream2>>>(k, n, Hp, rowSumsH);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));

    /*-----------------------------------------------------------------------------
     * compute numerator: (W*H)^(-1).*X*H^T m-x-k
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &cunmf_traits<T>::one, WHm1X, m, Hp, k, &cunmf_traits<T>::zero, WHm1XHT, m));

    /*-----------------------------------------------------------------------------
     * update W
     *-----------------------------------------------------------------------------*/
    {
        dim3 grid((m + 15) / 16, (k + 15) / 16);
        dim3 block(16, 16);
        if (opt->eps == 0.0) {
            MUWupdateKL<<<grid, block>>>(m, k, WHm1XHT, rowSumsH, Wp, W);
        } else {
            MUWupdateKL<<<grid, block>>>(static_cast<T>(opt->eps), m, k, WHm1XHT, rowSumsH, Wp, W);
        }
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
__global__ void colSum(int m, int k, const T *W, T *colSumsW) {
    int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    int stridej = blockDim.x * gridDim.x;
    for (int j = j0; j < k; j += stridej) {
        T sum = 0;
        for (int i = 0; i < m; i++) {
            sum += W[i + j * m];
        }
        colSumsW[j] = sum;
    }
}

template <typename T>
__global__ void MUHupdateKL(int k, int n, const T *numerator, const T *colSums, const T *Hp, T *H) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute H = Hp .* (numerator ./ (colsums + eps))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < n; j += stridej) {
        for (int i = i0; i < k; i += stridei) {
            H[i + j * k] = Hp[i + j * k] * numerator[i + j * k] / (colSums[i] + cunmf_traits<T>::eps);
        }
    }
}

template <typename T>
__global__ void MUHupdateKL(const T eps, int k, int n, const T *numerator, const T *colSums, const T *Hp, T *H) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute H = Hp .* (numerator ./ (colsums + eps))
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < n; j += stridej) {
        for (int i = i0; i < k; i += stridei) {
            H[i + j * k] = cunmf_traits<T>::maximum(eps, Hp[i + j * k] * numerator[i + j * k] / (colSums[i] + cunmf_traits<T>::eps));
        }
    }
}

template <typename T>
static int updateH_KL(cublasHandle_t cublasH, int m, int n, int k, const T *X, const T *W, const T *Hp, void *buffer, cunmf_options opt, T *H) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for
     *-----------------------------------------------------------------------------*/
    T *WHm1X = reinterpret_cast<T *>(buffer);              // (W*H).^(-1).*X
    T *WTWHm1X = reinterpret_cast<T *>(WHm1X + m * n);     // W^T*((W*H).^(-1).*X)
    T *colSumsW = reinterpret_cast<T *>(WTWHm1X + k * n);  // sum(W, 1)

    /*-----------------------------------------------------------------------------
     * compute: W*H m-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, Hp, k, &cunmf_traits<T>::zero, WHm1X, m));

    /*-----------------------------------------------------------------------------
     * compute: WH^(-1).*X on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    {
        dim3 grid((m + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        prepare_update_KL<<<grid, block, 0, stream1>>>(m, n, X, WHm1X);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * compute col sums of W on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    {
        dim3 grid((k + 31) / 32);
        dim3 block(32);
        colSum<<<grid, block, 0, stream2>>>(m, k, W, colSumsW);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));

    /*-----------------------------------------------------------------------------
     * compute numerator: W^T *(W*H)^(-1).*X k-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &cunmf_traits<T>::one, W, m, WHm1X, m, &cunmf_traits<T>::zero, WTWHm1X, k));

    /*-----------------------------------------------------------------------------
     * update H
     *-----------------------------------------------------------------------------*/
    {
        dim3 grid((k + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        if (opt->eps == 0.0) {
            MUHupdateKL<<<grid, block>>>(k, n, WTWHm1X, colSumsW, Hp, H);
        } else {
            MUHupdateKL<<<grid, block>>>(static_cast<T>(opt->eps), k, n, WTWHm1X, colSumsW, Hp, H);
        }
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
struct Betadiv : public thrust::binary_function<T, T, double> {
    T beta;
    T gamma;
    Betadiv(T beta) : beta(beta), gamma(static_cast<T>(1.0 / (beta * (beta - 1)))) {}

    __host__ __device__ double operator()(const T &x, const T &y) const {
        return gamma * (cunmf_traits<T>::power(x, beta) + (beta - 1) * cunmf_traits<T>::power(y, beta) - beta * x * cunmf_traits<T>::power(y, beta - 1));
    }
};

template <typename T>
static int cunmf_objbeta(cublasHandle_t cublasH, int m, int n, int k, double beta, const T *X, const T *W, const T *H,
                         void *buffer, double *obj) {
    /*-----------------------------------------------------------------------------
     * compute W*H
     *-----------------------------------------------------------------------------*/
    T *WH = reinterpret_cast<T *>(buffer);
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, H, k, &cunmf_traits<T>::zero, WH, m));

    /*-----------------------------------------------------------------------------
     * compute beta divergence
     *-----------------------------------------------------------------------------*/
    *obj = thrust::inner_product(thrust::device_pointer_cast(X), thrust::device_pointer_cast(X + m * n), thrust::device_pointer_cast(WH), 0.0, thrust::plus<double>(), Betadiv<T>(beta));

    return 0;
}

template <typename T>
__global__ void prepare_update_beta(int m, int n, double beta, const T *X, T *WHbetam1, T *WHbetam2X) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int stridei = blockDim.x * gridDim.x;
    int stridej = blockDim.y * gridDim.y;

    /*-----------------------------------------------------------------------------
     * compute (W*H).^(beta-1) and (W*H).^(beta-2).*X
     *-----------------------------------------------------------------------------*/
    for (int j = j0; j < n; j += stridej) {
        for (int i = i0; i < m; i += stridei) {
            T tmp = WHbetam1[i + j * m];
            WHbetam1[i + j * m] = cunmf_traits<T>::power(tmp, static_cast<T>(beta - 1));
            WHbetam2X[i + j * m] = cunmf_traits<T>::power(tmp, static_cast<T>(beta - 2)) * X[i + j * m];
        }
    }
}

template <typename T>
static int updateW_beta(cublasHandle_t cublasH, int m, int n, int k, double beta, const T *X, const T *Wp, const T *Hp, void *buffer, cunmf_options opt, T *W) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for
     *-----------------------------------------------------------------------------*/
    T *WHbetam1 = reinterpret_cast<T *>(buffer);                 // (W*H).^(beta-1)
    T *WHbetam2X = reinterpret_cast<T *>(WHbetam1 + m * n);      // (W*H).^(beta-2).*X
    T *WHbetam1HT = reinterpret_cast<T *>(WHbetam2X + m * n);    // (W*H).^(beta-1)*H^T
    T *WHbetam2XHT = reinterpret_cast<T *>(WHbetam1HT + m * k);  // (W*H).^(beta-2).*X*H^T

    /*-----------------------------------------------------------------------------
     * compute: W*H m-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, Wp, m, Hp, k, &cunmf_traits<T>::zero, WHbetam1, m));

    /*-----------------------------------------------------------------------------
     * compute: (W*H).^(beta-1) and (W*H).^(beta-2).*X m-x-n
     *-----------------------------------------------------------------------------*/
    {
        dim3 grid((m + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        prepare_update_beta<<<grid, block>>>(m, n, beta, X, WHbetam1, WHbetam2X);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * compute numerator: (W*H).^(-2).*X*H^T m-x-k on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream1));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &cunmf_traits<T>::one, WHbetam2X, m, Hp, k, &cunmf_traits<T>::zero, WHbetam2XHT, m));

    /*-----------------------------------------------------------------------------
     * compute denumerator: (W*H).^(-1)*H^T m-x-k on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream2));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &cunmf_traits<T>::one, WHbetam1, m, Hp, k, &cunmf_traits<T>::zero, WHbetam1HT, m));

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    /*-----------------------------------------------------------------------------
     * update W
     *-----------------------------------------------------------------------------*/
    dim3 grid((m + 15) / 16, (k + 15) / 16);
    dim3 block(16, 16);
    if (opt->eps == 0.0) {
        if (beta < 1.0) {
            MUupdate<<<grid, block>>>(m, k, static_cast<T>(1.0 / (2.0 - beta)), WHbetam2XHT, WHbetam1HT, Wp, W);
        } else if (beta >= 1.0 && beta <= 2.0) {
            MUupdate<<<grid, block>>>(m, k, WHbetam2XHT, WHbetam1HT, Wp, W);
        } else {
            MUupdate<<<grid, block>>>(m, k, static_cast<T>(1.0 / (beta - 1.0)), WHbetam2XHT, WHbetam1HT, Wp, W);
        }
    } else {
        if (beta < 1.0) {
            MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), m, k, static_cast<T>(1.0 / (2.0 - beta)), WHbetam2XHT, WHbetam1HT, Wp, W);
        } else if (beta >= 1.0 && beta <= 2.0) {
            MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), m, k, WHbetam2XHT, WHbetam1HT, Wp, W);
        } else {
            MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), m, k, static_cast<T>(1.0 / (beta - 1.0)), WHbetam2XHT, WHbetam1HT, Wp, W);
        }
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
static int updateH_beta(cublasHandle_t cublasH, int m, int n, int k, double beta, const T *X, const T *W, const T *Hp, void *buffer, cunmf_options opt, T *H) {
    /*-----------------------------------------------------------------------------
     * split memory buffer for
     *-----------------------------------------------------------------------------*/
    T *WHbetam1 = reinterpret_cast<T *>(buffer);                 // (W*H).^(beta-1)
    T *WHbetam2X = reinterpret_cast<T *>(WHbetam1 + m * n);      // (W*H).^(beta-2).*X
    T *WTWHbetam1 = reinterpret_cast<T *>(WHbetam2X + m * n);    // W^T*(W*H).^(beta-1)
    T *WTWHbetam2X = reinterpret_cast<T *>(WTWHbetam1 + k * n);  // W^T*(W*H).^(beta-2).*X

    /*-----------------------------------------------------------------------------
     * compute: W*H m-x-n
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cunmf_traits<T>::one, W, m, Hp, k, &cunmf_traits<T>::zero, WHbetam1, m));

    /*-----------------------------------------------------------------------------
     * compute: (W*H)^(beta-1) and (W*H)^(beta-2).*X
     *-----------------------------------------------------------------------------*/
    {
        dim3 grid((m + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        prepare_update_beta<<<grid, block>>>(m, n, beta, X, WHbetam1, WHbetam2X);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * compute numerator: W^T*(W*H).^(beta-2).*X k-x-n on stream 1
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream1));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &cunmf_traits<T>::one, W, m, WHbetam2X, m, &cunmf_traits<T>::zero, WTWHbetam2X, k));

    /*-----------------------------------------------------------------------------
     * compute denumerator: W^T*(W*H).^(-1) k-x-n on stream 2
     *-----------------------------------------------------------------------------*/
    cudaStream_t stream2;
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream2));
    CHECK_CUBLAS(cunmf_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &cunmf_traits<T>::one, W, m, WHbetam1, m, &cunmf_traits<T>::zero, WTWHbetam1, k));

    /*-----------------------------------------------------------------------------
     * synchronize streams and destroy them
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    /*-----------------------------------------------------------------------------
     * update W
     *-----------------------------------------------------------------------------*/
    dim3 grid((k + 15) / 16, (n + 15) / 16);
    dim3 block(16, 16);
    if (opt->eps == 0.0) {
        if (beta < 1.0) {
            MUupdate<<<grid, block>>>(k, n, static_cast<T>(1.0 / (2.0 - beta)), WTWHbetam2X, WTWHbetam1, Hp, H);
        } else if (beta >= 1.0 && beta <= 2.0) {
            MUupdate<<<grid, block>>>(k, n, WTWHbetam2X, WTWHbetam1, Hp, H);
        } else {
            MUupdate<<<grid, block>>>(k, n, static_cast<T>(1.0 / (beta - 1.0)), WTWHbetam2X, WTWHbetam1, Hp, H);
        }
    } else {
        if (beta < 1.0) {
            MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), k, n, static_cast<T>(1.0 / (2.0 - beta)), WTWHbetam2X, WTWHbetam1, Hp, H);
        } else if (beta >= 1.0 && beta <= 2.0) {
            MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), k, n, WTWHbetam2X, WTWHbetam1, Hp, H);
        } else {
            MUupdate<<<grid, block>>>(static_cast<T>(opt->eps), k, n, static_cast<T>(1.0 / (beta - 1.0)), WTWHbetam2X, WTWHbetam1, Hp, H);
        }
    }

    /*-----------------------------------------------------------------------------
     * check for errors and return
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template <typename T>
static int cunmf_MUbeta_buffersize(int m, int n, int k, double beta, size_t *bufferSize) {
    // additional space for holding Wp and Hp, max(space for objective computation, space for update W, space for update H)
    if (beta == 2.0) {
        *bufferSize = (m * k + k * n + std::max({m * n, 2 * m * k + k * k, 2 * k * n + k * k})) * sizeof(T);
    } else if (beta == 1.0) {
        *bufferSize = (m * k + k * n + std::max({m * n, m * n + m * k + k, m * n + k * n + k})) * sizeof(T);
    } else if (beta == 0.0) {
        *bufferSize = (m * k + k * n + std::max({m * n, 2 * m * n + 2 * m * k, 2 * m * n + 2 * k * n})) * sizeof(T);
    } else {
        *bufferSize = (m * k + k * n + std::max({m * n, 2 * m * n + 2 * m * k, 2 * m * n + 2 * k * n})) * sizeof(T);
    }
    return 0;
}

int cunmf_sMUbeta_buffersize(int m, int n, int k, double beta, size_t *bufferSize) {
    return cunmf_MUbeta_buffersize<float>(m, n, k, beta, bufferSize);
}

int cunmf_dMUbeta_buffersize(int m, int n, int k, double beta, size_t *bufferSize) {
    return cunmf_MUbeta_buffersize<double>(m, n, k, beta, bufferSize);
}

template <typename T>
int cunmf_MUbeta(int m, int n, int k, double beta, const T *X, void *buffer, const cunmf_options opt, T *W, T *H, cunmf_info info) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    double objective = 0.0, objectivep = 0.0, relchange_objective = 0.0;
    double diffW_Wp = 0.0, nrmW = 0.0;
    double diffH_Hp = 0.0, nrmH = 0.0;
    double relchange_WH = 0.0;
    T *Wp = NULL, *Hp = NULL;
    auto t0 = std::chrono::high_resolution_clock::now();
    cublasHandle_t cublasH;

    /*-----------------------------------------------------------------------------
     * check input arguments
     *-----------------------------------------------------------------------------*/
    if (opt == nullptr) {
        fprintf(stderr, "opt points to null.\n");
        fflush(stderr);
        return CUNMF_INVALID_ARGUMENT;
    }

    if (opt->eps < 0.0) {
        fprintf(stderr, "opt->eps is negative.\n");
        fflush(stderr);
        return CUNMF_INVALID_ARGUMENT;
    }

    if (info == nullptr) {
        fprintf(stderr, "info points to null.\n");
        fflush(stderr);
        return CUNMF_INVALID_ARGUMENT;
    }

    /*-----------------------------------------------------------------------------
     * allocate info
     *-----------------------------------------------------------------------------*/
    std::vector<double> objective_vec;
    std::vector<double> relchange_objective_vec;
    std::vector<double> relchange_WH_vec;
    std::vector<double> time_vec;

    /*-----------------------------------------------------------------------------
     * split memory buffer for Wp and Hp
     *-----------------------------------------------------------------------------*/
    Wp = reinterpret_cast<T *>(buffer);
    Hp = Wp + m * k;
    buffer = reinterpret_cast<void *>(Hp + k * n);
    std::swap(W, Wp);
    std::swap(H, Hp);

    /*-----------------------------------------------------------------------------
     * create cublas handle
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * evalute the objective
     *-----------------------------------------------------------------------------*/
    if (beta == 2.0) {
        CHECK_CUNMF(cunmf_objFro(cublasH, m, n, k, X, Wp, Hp, buffer, &objectivep));
    } else if (beta == 1.0) {
        CHECK_CUNMF(cunmf_objKL(cublasH, m, n, k, X, Wp, Hp, buffer, &objectivep));
    } else if (beta == 0.0) {
        CHECK_CUNMF(cunmf_objIS(cublasH, m, n, k, X, Wp, Hp, buffer, &objectivep));
    } else {
        CHECK_CUNMF(cunmf_objbeta(cublasH, m, n, k, beta, X, Wp, Hp, buffer, &objectivep));
    }

    /*-----------------------------------------------------------------------------
     * print status information header
     *-----------------------------------------------------------------------------*/
    if (opt->verbose && opt->maxiter > 0) {
        fprintf(stdout, "iteration |       time |    objective | rel. change objective | rel. change W and H\n");
        fprintf(stdout, "----------+------------+--------------+-----------------------+--------------------\n");
        fflush(stdout);
    }

    /*-----------------------------------------------------------------------------
     * start the main loop
     *-----------------------------------------------------------------------------*/
    for (int iter = 0; iter < opt->maxiter; ++iter) {
        /*-----------------------------------------------------------------------------
         * update W and H and evalute the objective
         *-----------------------------------------------------------------------------*/
        if (beta == 2.0) {
            CHECK_CUNMF(updateW_fro(cublasH, m, n, k, X, Wp, Hp, buffer, opt, W));
            CHECK_CUNMF(updateH_fro(cublasH, m, n, k, X, W, Hp, buffer, opt, H));
            CHECK_CUNMF(cunmf_objFro(cublasH, m, n, k, X, W, H, buffer, &objective));
        } else if (beta == 1.0) {
            CHECK_CUNMF(updateW_KL(cublasH, m, n, k, X, Wp, Hp, buffer, opt, W));
            CHECK_CUNMF(updateH_KL(cublasH, m, n, k, X, W, Hp, buffer, opt, H));
            CHECK_CUNMF(cunmf_objKL(cublasH, m, n, k, X, W, H, buffer, &objective));
        } else if (beta == 0.0) {
            CHECK_CUNMF(updateW_IS(cublasH, m, n, k, X, Wp, Hp, buffer, opt, W));
            CHECK_CUNMF(updateH_IS(cublasH, m, n, k, X, W, Hp, buffer, opt, H));
            CHECK_CUNMF(cunmf_objIS(cublasH, m, n, k, X, W, H, buffer, &objective));
        } else {
            CHECK_CUNMF(updateW_beta(cublasH, m, n, k, beta, X, Wp, Hp, buffer, opt, W));
            CHECK_CUNMF(updateH_beta(cublasH, m, n, k, beta, X, W, Hp, buffer, opt, H));
            CHECK_CUNMF(cunmf_objbeta(cublasH, m, n, k, beta, X, W, H, buffer, &objective));
        }

        relchange_objective = std::abs(objectivep - objective) / objective;

        /*-----------------------------------------------------------------------------
         * compute rel. change of W and H
         *-----------------------------------------------------------------------------*/
        CHECK_CUNMF(cunmf_diffnormFro(m, k, W, Wp, &diffW_Wp));
        CHECK_CUNMF(cunmf_normFro(m, k, W, &nrmW));
        CHECK_CUNMF(cunmf_diffnormFro(k, n, H, Hp, &diffH_Hp));
        CHECK_CUNMF(cunmf_normFro(k, n, H, &nrmH));
        relchange_WH = std::max(diffW_Wp / nrmW, diffH_Hp / nrmH);

        /*-----------------------------------------------------------------------------
         * measure time
         *-----------------------------------------------------------------------------*/
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9;

        /*-----------------------------------------------------------------------------
         * print status information
         *-----------------------------------------------------------------------------*/
        if (opt->verbose) {
            fprintf(stdout, "%9d | %*.2f | %e | %*e | %*e \n", iter + 1, 10, dt, objective, 21, relchange_objective, 18, relchange_WH);
            fflush(stdout);
        }

        /*-----------------------------------------------------------------------------
         * store info
         *-----------------------------------------------------------------------------*/
        objective_vec.push_back(objective);
        relchange_objective_vec.push_back(relchange_objective);
        relchange_WH_vec.push_back(relchange_WH);
        time_vec.push_back(dt);
        info->iter = iter;

        /*-----------------------------------------------------------------------------
         * check stopping criterions
         *-----------------------------------------------------------------------------*/
        if (dt >= opt->maxtime) {
            break;
        }

        if (diffW_Wp <= opt->tol_relchange_WH * nrmW && diffH_Hp <= opt->tol_relchange_WH * nrmH) {
            break;
        }

        if (std::abs(objectivep - objective) <= opt->tol_relchange_objective * objective) {
            break;
        }

        /*-----------------------------------------------------------------------------
         * swap W and Wp, H and Hp
         *-----------------------------------------------------------------------------*/
        std::swap(W, Wp);
        std::swap(H, Hp);

        /*-----------------------------------------------------------------------------
         * reset objectivep
         *-----------------------------------------------------------------------------*/
        objectivep = objective;
    }

    /*-----------------------------------------------------------------------------
     * store info
     *-----------------------------------------------------------------------------*/
    info->objective = reinterpret_cast<double *>(malloc(objective_vec.size() * sizeof(double)));
    memcpy(info->objective, objective_vec.data(), objective_vec.size() * sizeof(double));

    info->relchange_objective = reinterpret_cast<double *>(malloc(relchange_objective_vec.size() * sizeof(double)));
    memcpy(info->relchange_objective, relchange_objective_vec.data(), relchange_objective_vec.size() * sizeof(double));

    info->relchange_WH = reinterpret_cast<double *>(malloc(relchange_WH_vec.size() * sizeof(double)));
    memcpy(info->relchange_WH, relchange_WH_vec.data(), relchange_WH_vec.size() * sizeof(double));

    info->time = reinterpret_cast<double *>(malloc(time_vec.size() * sizeof(double)));
    memcpy(info->time, time_vec.data(), time_vec.size() * sizeof(double));

    /*-----------------------------------------------------------------------------
     * copy Wp and Hp to W and H if necessary
     *-----------------------------------------------------------------------------*/
    if (info->iter % 2 == 0) {
        CHECK_CUDA(cudaMemcpy(W, Wp, m * k * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(H, Hp, k * n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    /*-----------------------------------------------------------------------------
     * destroy cublas handle
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cublasDestroy(cublasH));

    return 0;
}

int cunmf_sMUbeta(int m, int n, int k, double beta, const float *X, void *buffer, const cunmf_options opt, float *W, float *H, cunmf_info info) {
    return cunmf_MUbeta(m, n, k, beta, X, buffer, opt, W, H, info);
}

int cunmf_dMUbeta(int m, int n, int k, double beta, const double *X, void *buffer, const cunmf_options opt, double *W, double *H, cunmf_info info) {
    return cunmf_MUbeta(m, n, k, beta, X, buffer, opt, W, H, info);
}
