# cuNMF - Nonnegative Matrix Factorization using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuNMF)
 [![DOI](https://zenodo.org/badge/760630733.svg)](https://zenodo.org/doi/10.5281/zenodo.10844444)



**Version:** 1.0.0

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuNMF` is a `CUDA` library implementing multiplicative update rules for the nonnegative matrix factorization $X\approx WH$, using the $\beta$-divergence as an error measure.

`cuNMF` supports real single and double precision matrices.

## Available Functions

### General Functions
```C
int cunmf_info_create(cunmf_info* info);
int cunmf_info_destroy(cunmf_info info);
```

### Single Precision Functions
```C
int cunmf_options_screate(cunmf_options* opt);
int cunmf_options_sdestroy(cunmf_options opt);

int cunmf_sMUbeta_buffersize(int m, int n, int k, double beta, size_t* bufferSize);
int cunmf_sMUbeta(int m, int n, int k, double beta, const float* X, void* buffer, const cunmf_options opt, float* W, float* H, cunmf_info info);
```

### Double Precision Functions
```C
int cunmf_options_dcreate(cunmf_options* opt);
int cunmf_options_ddestroy(cunmf_options opt);

int cunmf_dMUbeta_buffersize(int m, int n, int k, double beta, size_t* bufferSize);
int cunmf_dMUbeta(int m, int n, int k, double beta, const float* X, void* buffer, const cunmf_options opt, float* W, float* H, cunmf_info info);
```

## Algorithm

`cuNMF` implements the multiplicative update rules to minimize the $\beta$-divergence for the nonnegative matrix factorization $X\approx W H$, where
$X$ is of size $m\times n$, $W$ is if size $m\times k$ and $H$ is of size $k\times n$.

In more details, we consider the optimization problem

```math
   \min\limits_{W \geq \epsilon, H \geq \epsilon} D_{\beta}(X || WH)=\sum_{i=1}^{m}\sum_{j=1}^{n} d_{\beta}(X_{i,j},(WH)_{i,j})
```
where $\epsilon$ is a small nonegative constant and the $\beta$-divergence $d_{\beta}(x,y)$ is given by
```math
 d_{\beta}(x,y)=\left\{
  \begin{array}{ll}
  \frac{x}{y} -\log(\frac{x}{y}) -1                                            & \beta = 0,          \\
  x\log(\frac{x}{y}) -x + y                                                    & \beta = 1,          \\
  \frac{1}{\beta(\beta-1)}(x^{\beta} + (\beta-1)y^{\beta}-\beta xy^{\beta-1})  & \textrm{otherwise}.
 \end{array}
 \right.
```
The case $\beta=0$ gives the Itakura–Saito divergence, $\beta = 1$ gives the Kullback–Leibler divergence, and $\beta=2$ gives the Frobenius norm distance $\frac{1}{2}||\cdot||_F^2$.
For more details on the multiplicative update rule see Theorem 8.8 and Theorem 8.9 in 

> Gillis, Nicolas. _Nonnegative matrix factorization_. Society for Industrial and Applied Mathematics, 2020.


## Installation

Prerequisites:
 * `CMake >= 3.23`
 * `CUDA >= 11.4.2`

```shell
  mkdir build && cd build
  cmake ..
  make
  make install
```

## Usage and Examples

The multiplicate update algorithm is an iterative ones. The initial iterates $W_0$ and $H_0$ must be nonnegative. 
The parameter $k$ (number of columns of $W$ / rows of $H$) must be specified by the user. 
The user can also specifiy stopping criteria based on the 
 
* number of iterations (`maxiter`)
* computational time (`maxtime`)
* relative change of the iterates $W$ and $H$ (`tol_relchange_WH`)
* relative change of the objective $D_{\beta}$ (`tol_relchange_objective`).

See [`example_cunmf_MUbeta.cu`](example_cunmf_MUbeta.cu) for an example using double precision data.


