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

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
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

// see nn_batchNormMode_e
typedef enum
{
	NN_CODER_BATCH_NORM_MODE_NONE     = 0,
	NN_CODER_BATCH_NORM_MODE_RUNNING  = 1,
	NN_CODER_BATCH_NORM_MODE_INSTANCE = 2,
} nn_coderBatchNormMode_e;

#define NN_CODER_BATCH_NORMALIZATION_MODE_COUNT 3

typedef enum
{
	NN_CODER_OP_MODE_NONE         = 0,
	NN_CODER_OP_MODE_CONVT_2X2_S2 = 1, // upscale
	NN_CODER_OP_MODE_CONV_3X3_S2  = 2, // downscale
} nn_coderOpMode_e;

#define NN_CODER_OP_MODE_COUNT 5

typedef struct nn_coderLayerInfo_s
{
	nn_arch_t* arch;

	nn_dim_t* dimX;
	uint32_t  fc;

	// conv layer
	int      conv_flags;
	uint32_t conv_size;

	// skip layer
	// skip_coder must be set for add/cat modes
	nn_coderSkipMode_e skip_mode;
	nn_coderLayer_t*   skip_coder;
	float              skip_beta;

	// bn layer
	nn_coderBatchNormMode_e bn_mode;

	// fact layer
	nn_factLayerFn_e fact_fn;

	// op layer
	nn_coderOpMode_e op_mode;
} nn_coderLayerInfo_t;

typedef struct nn_coderOpLayer_s
{
	nn_layer_t base;

	nn_coderLayer_t* coder;

	nn_coderOpMode_e op_mode;

	// upscale layer
	// transpose, xavier, stride=2
	// W : dim(xd,2,2,xd)
	// Y : dim(bs,2*xh,2*xw,xd)
	//
	// downscale layer
	// xavier, stride=2
	// W : dim(xd,3,3,xd)
	// Y : dim(bs,xh/2,xw/2,xd)
	nn_convLayer_t* conv;
} nn_coderOpLayer_t;

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
	nn_coderOpLayer_t*   op;
} nn_coderLayer_t;

nn_coderLayer_t* nn_coderLayer_new(nn_coderLayerInfo_t* info);
void             nn_coderLayer_delete(nn_coderLayer_t** _self);
nn_coderLayer_t* nn_coderLayer_import(nn_arch_t* arch,
                                      jsmn_val_t* val,
                                      nn_coderLayer_t* skip_coder);
int              nn_coderLayer_export(nn_coderLayer_t* self,
                                      jsmn_stream_t* stream);
int              nn_coderLayer_lerp(nn_coderLayer_t* self,
                                    nn_coderLayer_t* lerp,
                                    float s1, float s2);

#endif
