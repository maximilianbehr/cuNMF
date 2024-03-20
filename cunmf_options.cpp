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

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "cunmf.h"
#include "cunmf_error.h"

template <typename T>
static int cunmf_options_create(cunmf_options* opt) {
    /*-----------------------------------------------------------------------------
     * check if input is nullpointer
     *-----------------------------------------------------------------------------*/
    if (opt == nullptr) {
        fprintf(stderr, "%s:%d opt points to NULL\n", __FILE__, __LINE__);
        fflush(stderr);
        return CUNMF_INVALID_ARGUMENT;
    }

    /*-----------------------------------------------------------------------------
     * allocate options struct
     *-----------------------------------------------------------------------------*/
    *opt = (cunmf_options)malloc(sizeof(cunmf_options_st));

    /*-----------------------------------------------------------------------------
     * check if allocation failed
     *-----------------------------------------------------------------------------*/
    if (*opt == nullptr) {
        fprintf(stderr, "%s:%d memory allocation failed\n", __FILE__, __LINE__);
        fflush(stderr);
        return CUNMF_INTERNAL_ERROR;
    }

    /*-----------------------------------------------------------------------------
     * set default options
     *-----------------------------------------------------------------------------*/
    (*opt)->eps = std::numeric_limits<T>::epsilon();
    (*opt)->maxiter = 1000;
    (*opt)->maxtime = 360;
    (*opt)->verbose = 1;
    (*opt)->tol_relchange_WH = 1e-10;
    (*opt)->tol_relchange_objective = 1e-10;

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return 0;
}

int cunmf_options_dcreate(cunmf_options* opt) {
    return cunmf_options_create<double>(opt);
}

int cunmf_options_screate(cunmf_options* opt) {
    return cunmf_options_create<float>(opt);
}

static int cunmf_options_destroy(cunmf_options opt) {
    /*-----------------------------------------------------------------------------
     * free memory
     *-----------------------------------------------------------------------------*/
    free(opt);

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return 0;
}

int cunmf_options_ddestroy(cunmf_options opt) {
    return cunmf_options_destroy(opt);
}

int cunmf_options_sdestroy(cunmf_options opt) {
    return cunmf_options_destroy(opt);
}
