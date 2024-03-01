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

#include "cunmf.h"
#include "cunmf_error.h"

int cunmf_info_create(cunmf_info* info) {
    /*-----------------------------------------------------------------------------
     * check if input is nullpointer
     *-----------------------------------------------------------------------------*/
    if (info == nullptr) {
        fprintf(stderr, "%s:%d info points to NULL\n", __FILE__, __LINE__);
        fflush(stderr);
        return CUNMF_INVALID_ARGUMENT;
    }

    /*-----------------------------------------------------------------------------
     * allocate info struct
     *-----------------------------------------------------------------------------*/
    *info = (cunmf_info)malloc(sizeof(cunmf_info_st));

    /*-----------------------------------------------------------------------------
     * check if allocation failed
     *-----------------------------------------------------------------------------*/
    if (*info == nullptr) {
        fprintf(stderr, "%s:%d memory allocation failed\n", __FILE__, __LINE__);
        fflush(stderr);
        return CUNMF_INTERNAL_ERROR;
    }

    /*-----------------------------------------------------------------------------
     * set default options
     *-----------------------------------------------------------------------------*/
    (*info)->iter = 0;
    (*info)->time = 0.0;
    (*info)->relchange_WH = nullptr;
    (*info)->relchange_objective = nullptr;
    (*info)->objective = nullptr;

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return 0;
}

int cunmf_info_destroy(cunmf_info info) {
    /*-----------------------------------------------------------------------------
     * free memory
     *-----------------------------------------------------------------------------*/
    free(info);

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return 0;
}
