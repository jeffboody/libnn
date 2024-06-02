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

#ifndef nn_lanczosResampler_H
#define nn_lanczosResampler_H

#include "nn.h"
#include "nn_dim.h"

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

void nn_lanczosParam_copy(nn_lanczosParam_t* src,
                          nn_lanczosParam_t* dst);

typedef struct nn_lanczosResampler_s
{
	nn_lanczosParam_t param;

	nn_dim_t dimX;
	nn_dim_t dimY;

	// Lanczos Resampling
	// * https://github.com/jeffboody/Lanczos
	// * intended only for data initialization
	// * CPU only and requires IO tensors
	// * does not support backprop
	// * see also nn_lanczosLayer_t
	// * only power-of-two resampling is supported
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
	nn_tensor_t* T;  // dim(1,xh,yw,1) (temp)
	nn_tensor_t* Lw; // dim(fcw,1,1,szw)
	nn_tensor_t* Lh; // dim(fch,1,1,szh)
} nn_lanczosResampler_t;

nn_lanczosResampler_t* nn_lanczosResampler_new(nn_engine_t* engine,
                                               nn_dim_t* dimX,
                                               nn_dim_t* dimY,
                                               int a);
void                   nn_lanczosResampler_delete(nn_lanczosResampler_t** _self);
int                    nn_lanczosResampler_resample(nn_lanczosResampler_t* self,
                                                    nn_tensor_t* X,
                                                    nn_tensor_t* Y,
                                                    uint32_t bs,
                                                    uint32_t xk,
                                                    uint32_t yk,
                                                    uint32_t depth);

#endif
