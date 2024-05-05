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

#ifndef nn_lanczosLayer_H
#define nn_lanczosLayer_H

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "../libvkk/vkk.h"
#include "nn_layer.h"

typedef struct nn_lanczosUs2Key_s
{
	uint32_t n;
} nn_lanczosUs2Key_t;

typedef struct nn_lanczosUs2Data_s
{
	vkk_buffer_t*     sb200;
	vkk_uniformSet_t* us2;
} nn_lanczosUs2Data_t;

nn_lanczosUs2Data_t* nn_lanczosUs2Data_new(nn_engine_t* engine,
                                           nn_lanczosUs2Key_t* key);
void                 nn_lanczosUs2Data_delete(nn_lanczosUs2Data_t** _self);

typedef struct nn_lanczosParam_s
{
	int a;
	int fsw;
	int fsh;
	int fcw;
	int fch;
	int szw;
	int szh;
} nn_lanczosParam_t;

typedef struct nn_lanczosLayer_s
{
	nn_layer_t base;

	nn_lanczosParam_t param;

	// Lanczos Resampling
	// * https://github.com/jeffboody/Lanczos
	// * only power-of-two resampling is supported
	// * always use same padding (zero outside bounds)
	// * support size (a)
	// * filter scale
	//   fsw = xw/yw
	//   fsh = xh/yh
	//   if(fsw < 1) fsw = 1;
	//   if(fsh < 1) fsh = 1;
	// * filter count
	//   fcw = yw/xw
	//   fch = yh/xh
	//   if(fcw < 1) fcw = 1
	//   if(fch < 1) fch = 1
	// * filter size
	//   szw = 2*fsw*a
	//   szh = 2*fsh*a
	//
	// W: width "separable" pass output
	// Y: output
	// Lw/Lh: Lanczos kernels (precomputed and premultiplied)
	nn_tensor_t* X;  // dim(bs,xh,xw,xd) (reference)
	nn_tensor_t* T;  // dim(bs,xh,yw,xd) (temp)
	nn_tensor_t* Y;  // dim(bs,yh,yw,xd)
	nn_tensor_t* Lw; // dim(fcw,1,1,szw)
	nn_tensor_t* Lh; // dim(fch,1,1,szh)

	// forward gradients
	// dT_dX; // Lw : dim(fcw,1,1,szw)
	// dY_dT; // Lh : dim(fch,1,1,szh)

	// backprop gradients
	//           dL_dY; // dim(bs,yh,yw,xd)
	nn_tensor_t* dL_dT; // dim(bs,xh,yw,xd)
	nn_tensor_t* dL_dX; // dim(bs,xh,xw,xd)

	vkk_buffer_t*     sb008_param;
	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1_fp;
	vkk_uniformSet_t* us1_bp;
} nn_lanczosLayer_t;

nn_lanczosLayer_t* nn_lanczosLayer_new(nn_arch_t* arch,
                                       nn_dim_t* dimX,
                                       nn_dim_t* dimY,
                                       int a);
void               nn_lanczosLayer_delete(nn_lanczosLayer_t** _self);
nn_lanczosLayer_t* nn_lanczosLayer_import(nn_arch_t* arch,
                                          jsmn_val_t* val);
int                nn_lanczosLayer_export(nn_lanczosLayer_t* self,
                                          jsmn_stream_t* stream);

#endif
