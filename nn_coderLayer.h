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
#include "../libcc/cc_list.h"
#include "nn_dim.h"
#include "nn_layer.h"

typedef enum
{
	NN_CODER_CONV_MODE_NONE     = 0,
	NN_CODER_CONV_MODE_3X3_RELU = 1,
} nn_coderConvMode_e;

#define NN_CODER_CONV_MODE_COUNT 2

typedef enum
{
	NN_CODER_SKIP_MODE_NONE = 0,
	NN_CODER_SKIP_MODE_FORK = 1,
	NN_CODER_SKIP_MODE_ADD  = 2,
	NN_CODER_SKIP_MODE_CAT  = 3,
} nn_coderSkipMode_e;

#define NN_CODER_SKIP_MODE_COUNT 4

// see nn_batchNormMode_e
// RUNNING is default
typedef enum
{
	NN_CODER_BATCH_NORM_MODE_NONE     = -1,
	NN_CODER_BATCH_NORM_MODE_RUNNING  = 0,
	NN_CODER_BATCH_NORM_MODE_INSTANCE = 1,
} nn_coderBatchNormMode_e;

#define NN_CODER_BATCH_NORMALIZATION_MODE_COUNT 2

// see nn_tensorNormMode_e
// NONE is default
typedef enum
{
	NN_CODER_NORM_MODE_NONE = 0,
	NN_CODER_NORM_MODE_SN   = 1,
	NN_CODER_NORM_MODE_BSSN = 2,
} nn_coderTensorNormMode_e;

#define NN_CODER_NORM_MODE_COUNT 3

typedef enum
{
	NN_CODER_OP_MODE_NONE         = 0,
	NN_CODER_OP_MODE_CONVT_2X2_S2 = 1, // upscale
	NN_CODER_OP_MODE_CONVT_6X6_S2 = 2, // upscale
	NN_CODER_OP_MODE_CONV_3X3_S2  = 3, // downscale
	NN_CODER_OP_MODE_POOL_MAX_S2  = 4,
	NN_CODER_OP_MODE_POOL_AVG_S2  = 5,
} nn_coderOpMode_e;

#define NN_CODER_OP_MODE_COUNT 6

typedef struct nn_coderLayerInfo_s
{
	nn_arch_t* arch;

	nn_dim_t* dimX;
	uint32_t  fc;

	// tensor normalization
	nn_coderTensorNormMode_e norm_mode;

	// pre operation layer
	nn_coderOpMode_e pre_op_mode;

	// conv layer
	nn_coderConvMode_e conv_mode;

	// skip layer
	// skip_coder must be set for add/cat modes
	nn_coderSkipMode_e skip_mode;
	nn_coderLayer_t*   skip_coder;

	// bn layer
	nn_coderBatchNormMode_e bn_mode;

	// repeater layers
	nn_coderConvMode_e repeat_mode;
	uint32_t           repeat;

	// post operation layer
	nn_coderOpMode_e post_op_mode;
} nn_coderLayerInfo_t;

typedef struct nn_coderOpLayer_s
{
	nn_layer_t base;

	nn_coderLayer_t* coder;

	nn_coderOpMode_e op_mode;
	union
	{
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

		// pooling layer
		// 2x2, max or avg
		// Y : dim(bs,xh/2,xw/2,xd)
		nn_poolingLayer_t* pool;
	};
} nn_coderOpLayer_t;

typedef struct nn_coderLayer_s
{
	nn_layer_t base;

	nn_dim_t dimX;
	nn_dim_t dimY;

	// tensor normalization
	nn_coderTensorNormMode_e norm_mode;

	// layers may be NULL depending on the desired modes

	// pre operation layer
	nn_coderOpLayer_t* pre_op;

	// main layer
	// disable_bias, he, relu
	// W : dim(fc,3,3,xd)
	// Y : dim(bs,xh,xw,fc)
	//
	// skip layer
	// fork (encoder)
	// add/cat (decoder)
	nn_convLayer_t*      conv;
	nn_skipLayer_t*      skip;
	nn_batchNormLayer_t* bn;
	nn_factLayer_t*      fact;

	// repeater layers
	cc_list_t* repeater;

	// post operation layer
	nn_coderOpLayer_t* post_op;
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
