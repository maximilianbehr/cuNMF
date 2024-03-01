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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cunmf_info_st {
    int iter;
    double* relchange_WH;
    double* relchange_objective;
    double* objective;
    double time;
} cunmf_info_st;

typedef cunmf_info_st* cunmf_info;

int cunmf_info_create(cunmf_info* info);
int cunmf_info_destroy(cunmf_info info);

typedef struct cunmf_options_st {
    int maxiter;
    double maxtime;
    int verbose;
    double tol_relchange_WH;
    double tol_relchange_objective;
} cunmf_options_st;

typedef cunmf_options_st* cunmf_options;

int cunmf_options_create(cunmf_options* opt);
int cunmf_options_destroy(cunmf_options opt);

#ifdef __cplusplus
}
#endif
