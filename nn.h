/*
 * Copyright (c) 2023 Jeff Boody
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef nn_H
#define nn_H

#include <stdint.h>

typedef struct nn_arch_s               nn_arch_t;
typedef struct nn_batchNormLayer_s     nn_batchNormLayer_t;
typedef struct nn_convLayer_s          nn_convLayer_t;
typedef struct nn_dim_s                nn_dim_t;
typedef struct nn_coderLayer_s         nn_coderLayer_t;
typedef struct nn_coderOpLayer_s       nn_coderOpLayer_t;
typedef struct nn_coderRepeaterLayer_s nn_coderRepeaterLayer_t;
typedef struct nn_factLayer_s          nn_factLayer_t;
typedef struct nn_flattenLayer_s       nn_flattenLayer_t;
typedef struct nn_layer_s              nn_layer_t;
typedef struct nn_loss_s               nn_loss_t;
typedef struct nn_poolingLayer_s       nn_poolingLayer_t;
typedef struct nn_skipLayer_s          nn_skipLayer_t;
typedef struct nn_tensor_s             nn_tensor_t;
typedef struct nn_weightLayer_s        nn_weightLayer_t;

#endif
