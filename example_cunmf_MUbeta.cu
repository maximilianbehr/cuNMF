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

#include <stdio.h>
#include <stdlib.h>

#include "cunmf.h"

int main(void) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    const int m = 10000, n = 500;  // size of the input matrix X
    const int k = 40;              // size of the factor matrices W m-by-k and H k-by-n
    double *X, *W, *H;             // input matrix and factor matrices
    void *dbuffer;                 // device buffer
    double beta = 2.0;             // minimize ||X - W*H||_F^2
    // const double beta = 1.0;  // minimize Kullback-Leibler divergence
    // const double beta = 0.0;  // minimize Itakura-Saito divergence
    // const double beta = 0.5;  // minimize general beta-divergence

    /*-----------------------------------------------------------------------------
     * allocate X, W, and H on the host
     *-----------------------------------------------------------------------------*/
    cudaMallocManaged((void **)&X, sizeof(*X) * m * n);
    cudaMallocManaged((void **)&W, sizeof(*W) * m * k);
    cudaMallocManaged((void **)&H, sizeof(*H) * k * n);

    /*-----------------------------------------------------------------------------
     * read nonnegative matrix X and nonnegative initial matrices W, H from file
     * all input matrices are stored in column-major order and must be nonnegative
     *-----------------------------------------------------------------------------*/
    FILE *fileX = fopen("X_10000_500.bin", "rb");
    fread(X, sizeof(*X), m * n, fileX);
    fclose(fileX);
    FILE *fileW0 = fopen("W0_10000_40.bin", "rb");
    fread(W, sizeof(*W), m * k, fileW0);
    fclose(fileW0);
    FILE *fileH0 = fopen("H0_40_500.bin", "rb");
    fread(H, sizeof(*H), k * n, fileH0);
    fclose(fileH0);

    /*-----------------------------------------------------------------------------
     * perform a workspace query and allocate memory buffer on the device
     *-----------------------------------------------------------------------------*/
    size_t bufferSize = 0;
    cunmf_dMUbeta_buffersize(m, n, k, beta, &bufferSize);  // use cunmf_sMUbeta_buffersize for single precision
    cudaMalloc((void **)&dbuffer, bufferSize);

    /*-----------------------------------------------------------------------------
     * create options struct
     *-----------------------------------------------------------------------------*/
    cunmf_options opt;
    cunmf_options_dcreate(&opt);           // use cunmf_options_screate for single precision
    opt->maxiter = 10;                     // maximum number of iterations
    opt->maxtime = 60.0;                   // maximum compute time in seconds
    opt->verbose = 1;                      // verbosity level
    opt->eps = 1e-16;                      // enforce nonnegative by W, H >= eps elementwise
    opt->tol_relchange_WH = 1e-10;         // tolerance for relative change of W and H
    opt->tol_relchange_objective = 1e-10;  // tolerance for relative change of the objective function

    /*-----------------------------------------------------------------------------
     * call cunmf_dMUbeta to compute W and H and print results
     *-----------------------------------------------------------------------------*/
    cunmf_info info;
    cunmf_info_create(&info);
    cunmf_dMUbeta(m, n, k, beta, X, dbuffer, opt, W, H, info);  // use cunmf_sMUbeta for single precision
    printf("beta=%.2f, cunmf_dMUbeta finished in %f seconds after %d iterations with objective value = %e\n", beta, info->time[info->iter], info->iter + 1, info->objective[info->iter]);

    /*-----------------------------------------------------------------------------
     * clear memory
     *-----------------------------------------------------------------------------*/
    cunmf_info_destroy(info);
    cunmf_options_ddestroy(opt);  // use cunmf_options_sdestroy for single precision
    cudaFree(dbuffer);
    cudaFree(H);
    cudaFree(W);
    cudaFree(X);
    return 0;
}
