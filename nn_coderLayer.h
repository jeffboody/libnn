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

#define NN_CODER_OP_MODE_NONE      0
#define NN_CODER_OP_MODE_UPSCALE   1
#define NN_CODER_OP_MODE_DOWNSCALE 2
#define NN_CODER_OP_MODE_POOLMAX   3
#define NN_CODER_OP_MODE_POOLAVG   4

typedef struct nn_coderLayerInfo_s
{
	nn_arch_t* arch;

	nn_dim_t* dimX;
	uint32_t  fc;

	// skip layer (optional)
	// skip_coder must be set for add/cat modes
	int              skip_enable;
	int              skip_mode;
	nn_coderLayer_t* skip_coder;

	// repeater layers (optional)
	uint32_t repeat;

	// operation layer (optional)
	int op_mode;
} nn_coderLayerInfo_t;

typedef struct nn_coderLayer_s
{
	nn_layer_t base;

	// main layer
	// disable_bias, he, relu
	// W : dim(fc,3,3,xd)
	// Y : dim(bs,xh,xw,fc)
	//
	// skip layer (optional)
	// fork (encoder)
	// add/cat (decoder)
	nn_convLayer_t*      conv;
	nn_skipLayer_t*      skip;
	nn_batchNormLayer_t* bn;
	nn_factLayer_t*      fact;

	// repeater layers (optional)
	cc_list_t* repeater;

	// operation layer (optional)
	nn_coderOpLayer_t* op;
} nn_coderLayer_t;

nn_coderLayer_t* nn_coderLayer_new(nn_coderLayerInfo_t* info);
void             nn_coderLayer_delete(nn_coderLayer_t** _self);
nn_coderLayer_t* nn_coderLayer_import(nn_arch_t* arch,
                                      jsmn_val_t* val,
                                      nn_coderLayer_t* skip_coder);
int              nn_coderLayer_export(nn_coderLayer_t* self,
                                      jsmn_stream_t* stream);

#endif
