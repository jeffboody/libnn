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

#ifndef nn_coderLayer_H
#define nn_coderLayer_H

#include "../libcc/jsmn/cc_jsmnStream.h"
#include "../libcc/jsmn/cc_jsmnWrapper.h"
#include "nn_dim.h"
#include "nn_factLayer.h"
#include "nn_layer.h"

// see nn_skipMode_e
typedef enum
{
	NN_CODER_SKIP_MODE_NONE     = 0,
	NN_CODER_SKIP_MODE_FORK_ADD = 1,
	NN_CODER_SKIP_MODE_FORK_CAT = 2,
	NN_CODER_SKIP_MODE_ADD      = 3,
	NN_CODER_SKIP_MODE_CAT      = 4,
} nn_coderSkipMode_e;

#define NN_CODER_SKIP_MODE_COUNT 5

typedef enum
{
	NN_CODER_BATCH_NORM_MODE_DISABLE = 0,
	NN_CODER_BATCH_NORM_MODE_ENABLE  = 1,
} nn_coderBatchNormMode_e;

typedef struct nn_coderLayerInfo_s
{
	nn_arch_t* arch;

	nn_dim_t* dimX;
	uint32_t  fc;

	// conv layer
	int      conv_flags;
	uint32_t conv_size;
	uint32_t conv_stride;

	// skip layer
	// skip_coder must be set for add/cat modes
	nn_coderSkipMode_e skip_mode;
	nn_coderLayer_t*   skip_coder;
	float              skip_beta;

	// bn layer
	nn_coderBatchNormMode_e bn_mode;

	// fact layer
	nn_factLayerFn_e fact_fn;
} nn_coderLayerInfo_t;

// The coderLayer is a composite layer consisting of batch
// normalization, activation, convolution and skip layers.
// It is intended to be used as a building block for
// encoder/decoder architectures such as U-Net and RRDB
// (residual-in-residual dense blocks). The coder layer
// helps to properly order the layers using the best known
// practices. The coder layer order was largely influenced
// by the paper "Identity Mappings in Deep Residual
// Networks" which determined that the original Residual
// Unit were improperly ordered resulting large errors. The
// findings in the Identity Mappings paper were specific to
// the add skip connection. The "Densely Connected
// Convolutional Networks" paper used the same layer order
// with the concatenation skip connection, however, I
// noticed that this leads to a significant performance
// overhead due to redundant application of batch
// normalization and activation layers with the RRDB
// architecture. As a result, the implementation changes
// the skip layer placement depending on if an add or
// concatenate operation is desired. When batch
// normalization is enabled and the skip connection is not
// an add operation then the conv flag is set to disable the
// redundant bias. The convolution layer initialization flag
// is set automatically (He or Xavier) depending on the type
// of activation function that is used. Note that the order
// of layers may appear different from the Identity Mappings
// paper at first glance. However, the proposed pattern of
// layers becomes apparent when the coder layer is chained
// together multiple times.
typedef struct nn_coderLayer_s
{
	nn_layer_t base;

	nn_dim_t dimX;
	nn_dim_t dimY;

	// layers may be NULL
	// skip attachment order
	//   FORK_ADD/ADD: after conv
	//   FORK_CAT/CAT: after fact
	nn_convLayer_t*      conv;
	nn_skipLayer_t*      skip;
	nn_batchNormLayer_t* bn;
	nn_factLayer_t*      fact;
} nn_coderLayer_t;

nn_coderLayer_t* nn_coderLayer_new(nn_coderLayerInfo_t* info);
void             nn_coderLayer_delete(nn_coderLayer_t** _self);
nn_coderLayer_t* nn_coderLayer_import(nn_arch_t* arch,
                                      cc_jsmnVal_t* val,
                                      nn_coderLayer_t* skip_coder);
int              nn_coderLayer_export(nn_coderLayer_t* self,
                                      cc_jsmnStream_t* stream);

#endif
