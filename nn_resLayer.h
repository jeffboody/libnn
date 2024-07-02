/*
 * Copyright (c) 2024 Jeff Boody
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

#ifndef nn_resLayer_H
#define nn_resLayer_H

#include "../libcc/jsmn/cc_jsmnStream.h"
#include "../libcc/jsmn/cc_jsmnWrapper.h"
#include "nn_layer.h"

typedef enum
{
	NN_RES_BATCH_NORM_MODE_DISABLE = 0,
	NN_RES_BATCH_NORM_MODE_ENABLE  = 1,
} nn_resBatchNormMode_e;

typedef struct nn_resLayer_s
{
	nn_layer_t base;

	// Identity Mappings in Deep Residual Networks
	// https://arxiv.org/pdf/1603.05027
	nn_skipLayer_t*      skip1;
	nn_batchNormLayer_t* bn1;
	nn_factLayer_t*      fact1;
	nn_convLayer_t*      conv1;
	nn_batchNormLayer_t* bn2;
	nn_factLayer_t*      fact2;
	nn_convLayer_t*      conv2;
	nn_skipLayer_t*      skip2;
} nn_resLayer_t;

nn_resLayer_t* nn_resLayer_new(nn_arch_t* arch,
                               nn_dim_t* dimX,
                               float skip_beta,
                               nn_resBatchNormMode_e bn_mode,
                               nn_factLayerFn_e fact_fn,
                               int norm_flags);
void           nn_resLayer_delete(nn_resLayer_t** _self);
nn_resLayer_t* nn_resLayer_import(nn_arch_t* arch,
                                  cc_jsmnVal_t* val);
int            nn_resLayer_export(nn_resLayer_t* self,
                                  cc_jsmnStream_t* stream);

#endif
