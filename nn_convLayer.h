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

#ifndef nn_convLayer_H
#define nn_convLayer_H

#include "nn_layer.h"

// XAVIER and VALID are default
#define NN_CONV_LAYER_FLAG_XAVIER       0x0001
#define NN_CONV_LAYER_FLAG_HE           0x0002
#define NN_CONV_LAYER_FLAG_DISABLE_BIAS 0x0010
#define NN_CONV_LAYER_FLAG_PAD_VALID    0x0100
#define NN_CONV_LAYER_FLAG_PAD_SAME     0x0200
#define NN_CONV_LAYER_FLAG_TRANSPOSE    0x1000

typedef struct nn_convLayer_s
{
	nn_layer_t base;

	int flags;

	uint32_t stride;

	// weights, bias, output
	// s  = stride
	// Standard
	//   yh = ((xh - fh)/s + 1) (valid) or xh/s (same)
	//   yw = ((xw - fw)/s + 1) (valid) or xw/s (same)
	// Transpose
	//   yh = s*(xh - 1) + fh (valid) or s*xh (same)
	//   yw = s*(xw - 1) + fw (valid) or s*xw (same)
	nn_tensor_t* X; // dim(bs,xh,xw,xd) (reference)
	nn_tensor_t* W; // dim(fc,fh,fw,xd)
	nn_tensor_t* B; // dim(fc,1,1,1)
	nn_tensor_t* Y; // dim(bs,yh,yw,fc)

	// momentum update
	nn_tensor_t* VW; // dim(fc,fh,fw,xd)
	nn_tensor_t* VB; // dim(fc,1,1,1)

	// gradient clipping
	float norm_dl_dw_ra;
	float norm_dl_db_ra;

	// forward gradients
	// dY_dB; // 1
	// dY_dX; // W : dim(fc,fh,fw,xd)
	// dY_dW; // X : dim(bs,xh,xw,xd)

	// backprop gradients
	//           dL_dY; // dim(bs,yh,yw,fc)
	nn_tensor_t* dL_dW; // dim(fc,fh,fw,xd)
	nn_tensor_t* dL_dB; // dim(fc,1,1,1)
	nn_tensor_t* dL_dX; // dim(bs,xh,xw,xd)
} nn_convLayer_t;

nn_convLayer_t* nn_convLayer_new(nn_arch_t* arch,
                                 nn_dim_t* dimX,
                                 nn_dim_t* dimW,
                                 uint32_t stride,
                                 int flags);
nn_convLayer_t* nn_convLayer_import(nn_arch_t* arch,
                                    jsmn_val_t* val);
int             nn_convLayer_export(nn_convLayer_t* self,
                                    jsmn_stream_t* stream);
void            nn_convLayer_delete(nn_convLayer_t** _self);

#endif
