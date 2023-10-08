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

#ifndef nn_poolingLayer_H
#define nn_poolingLayer_H

#include "../libvkk/vkk.h"
#include "nn_layer.h"

typedef enum
{
	NN_POOLING_MODE_ERROR   = -1,
	NN_POOLING_MODE_MAX     = 0,
	NN_POOLING_MODE_AVERAGE = 1,
} nn_poolingMode_e;

#define NN_POOLING_MODE_COUNT 2

typedef struct nn_poolingLayer_s
{
	nn_layer_t base;

	uint32_t stride;

	nn_poolingMode_e pooling_mode;

	// output
	// yh = xh/stride
	// yw = xw/stride
	//           X; // dim(bs,xh,xw,xd)
	nn_tensor_t* Y; // dim(bs,yh,yw,xd)

	// forward gradients
	nn_tensor_t* dY_dX; // dim(bs,xh,xw,xd)

	// backprop gradients
	//           dL_dY; // dim(bs,yh,yw,xd)
	nn_tensor_t* dL_dX; // dim(bs,xh,xw,xd)

	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1;
	vkk_uniformSet_t* us2;
	vkk_buffer_t*     sb01_param;
} nn_poolingLayer_t;

nn_poolingLayer_t* nn_poolingLayer_new(nn_arch_t* arch,
                                       nn_dim_t* dimX,
                                       uint32_t stride,
                                       nn_poolingMode_e pooling_mode);
nn_poolingLayer_t* nn_poolingLayer_import(nn_arch_t* arch,
                                          jsmn_val_t* val);
int                nn_poolingLayer_export(nn_poolingLayer_t* self,
                                          jsmn_stream_t* stream);
void               nn_poolingLayer_delete(nn_poolingLayer_t** _self);

#endif
