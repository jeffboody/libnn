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

#ifndef nn_lanczos3Layer_H
#define nn_lanczos3Layer_H

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "../libvkk/vkk.h"
#include "nn_layer.h"

typedef struct nn_lanczos3Us2Key_s
{
	uint32_t n;
} nn_lanczos3Us2Key_t;

typedef struct nn_lanczos3Us2Data_s
{
	vkk_buffer_t*     sb200;
	vkk_uniformSet_t* us2;
} nn_lanczos3Us2Data_t;

nn_lanczos3Us2Data_t* nn_lanczos3Us2Data_new(nn_engine_t* engine,
                                             nn_lanczos3Us2Key_t* key);
void                  nn_lanczos3Us2Data_delete(nn_lanczos3Us2Data_t** _self);

typedef struct nn_lanczos3Layer_s
{
	nn_layer_t base;

	int level;

	// lanczos3 filter (see texgz_tex_lanczos3)
	// separable filter, weights and output
	// always use same padding
	// int   scale   = cc_pow2n(level); // aka stride
	// float support = 3.0f;
	// float scalef  = (float) scale;
	// int   n       = (int) (scalef*support + 0.01f);
	// int   sz      = 2*n;
	// yh = xh/scale
	// yw = xw/scale
	nn_tensor_t* X; // dim(bs,xh,xw,xd) (reference)
	nn_tensor_t* H; // dim(bs,xh,yw,xd)
	nn_tensor_t* W; // dim(1,1,1,sz)
	nn_tensor_t* Y; // dim(bs,yh,yw,xd)

	// forward gradients
	// dH_dX; // W : dim(1,1,1,sz)
	// dY_dH; // W : dim(1,1,1,sz)

	// backprop gradients
	//           dL_dY; // dim(bs,yh,yw,xd)
	nn_tensor_t* dL_dH; // dim(bs,xh,yw,xd)
	nn_tensor_t* dL_dX; // dim(bs,xh,xw,xd)

	vkk_buffer_t*     sb008_param;
	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1_fp;
	vkk_uniformSet_t* us1_bp;
} nn_lanczos3Layer_t;

nn_lanczos3Layer_t* nn_lanczos3Layer_new(nn_arch_t* arch,
                                         nn_dim_t* dimX,
                                         int level);
void                nn_lanczos3Layer_delete(nn_lanczos3Layer_t** _self);
nn_lanczos3Layer_t* nn_lanczos3Layer_import(nn_arch_t* arch,
                                            jsmn_val_t* val);
int                 nn_lanczos3Layer_export(nn_lanczos3Layer_t* self,
                                            jsmn_stream_t* stream);

#endif
