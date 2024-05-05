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
typedef struct nn_batchNormUs2Data_s   nn_batchNormUs2Data_t;
typedef struct nn_batchNormUs2Key_s    nn_batchNormUs2Key_t;
typedef struct nn_convLayer_s          nn_convLayer_t;
typedef struct nn_convUs2Data_s        nn_convUs2Data_t;
typedef struct nn_convUs2Key_s         nn_convUs2Key_t;
typedef struct nn_coderLayerInfo_s     nn_coderLayerInfo_t;
typedef struct nn_coderLayer_s         nn_coderLayer_t;
typedef struct nn_dim_s                nn_dim_t;
typedef struct nn_engine_s             nn_engine_t;
typedef struct nn_factLayer_s          nn_factLayer_t;
typedef struct nn_reshapeLayer_s       nn_reshapeLayer_t;
typedef struct nn_lanczosLayer_s       nn_lanczosLayer_t;
typedef struct nn_lanczosParam_s       nn_lanczosParam_t;
typedef struct nn_layerInfo_s          nn_layerInfo_t;
typedef struct nn_layer_s              nn_layer_t;
typedef struct nn_loss_s               nn_loss_t;
typedef struct nn_skipLayer_s          nn_skipLayer_t;
typedef struct nn_tensorOpKUs0Idx_s    nn_tensorOpKUs0Idx_t;
typedef struct nn_tensorOpKUs0Data_s   nn_tensorOpKUs0Data_t;
typedef struct nn_tensorStats_s        nn_tensorStats_t;
typedef struct nn_tensor_s             nn_tensor_t;
typedef struct nn_urrdbBlockLayer_s    nn_urrdbBlockLayer_t;
typedef struct nn_urrdbLayerInfo_s     nn_urrdbLayerInfo_t;
typedef struct nn_urrdbLayer_s         nn_urrdbLayer_t;
typedef struct nn_urrdbNodeLayer_s     nn_urrdbNodeLayer_t;
typedef struct nn_weightLayer_s        nn_weightLayer_t;

#endif
