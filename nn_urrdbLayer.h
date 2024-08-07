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

#ifndef nn_urrdbLayer_H
#define nn_urrdbLayer_H

#include "../libcc/jsmn/cc_jsmnStream.h"
#include "../libcc/jsmn/cc_jsmnWrapper.h"
#include "../libcc/cc_list.h"
#include "nn_coderLayer.h"
#include "nn_dim.h"
#include "nn_layer.h"

typedef struct nn_urrdbLayerInfo_s
{
	nn_arch_t* arch;

	// blocks: number of dense blocks
	// nodes:  number of nodes per block (nodes >= 2)
	nn_dim_t* dimX;
	uint32_t  fc;
	uint32_t  blocks;
	uint32_t  nodes;

	// begin/end
	int                     norm_flags0;
	uint32_t                conv_size0;
	float                   skip_beta0;
	nn_coderBatchNormMode_e bn_mode0;
	nn_factLayerFn_e        fact_fn0;

	// dense blocks/nodes
	int                     norm_flags1;
	uint32_t                conv_size1;
	float                   skip_beta1; // add only
	nn_coderBatchNormMode_e bn_mode1;
	nn_factLayerFn_e        fact_fn1;
} nn_urrdbLayerInfo_t;

// Unified Residual-in-Residual Dense Block
typedef struct nn_urrdbLayer_s
{
	nn_layer_t base;

	nn_coderLayer_t* coder0;
	cc_list_t*       blocks;
	nn_coderLayer_t* coder1;
	nn_coderLayer_t* coder2;
} nn_urrdbLayer_t;

nn_urrdbLayer_t* nn_urrdbLayer_new(nn_urrdbLayerInfo_t* info);
void             nn_urrdbLayer_delete(nn_urrdbLayer_t** _self);
nn_urrdbLayer_t* nn_urrdbLayer_import(nn_arch_t* arch,
                                      cc_jsmnVal_t* val);
int              nn_urrdbLayer_export(nn_urrdbLayer_t* self,
                                      cc_jsmnStream_t* stream);

#endif
