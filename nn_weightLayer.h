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

#ifndef nn_weightLayer_H
#define nn_weightLayer_H

#include "../libvkk/vkk.h"
#include "nn_layer.h"

// XAVIER is default
#define NN_WEIGHT_LAYER_FLAG_XAVIER       0x0001
#define NN_WEIGHT_LAYER_FLAG_HE           0x0002
#define NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS 0x0010
#define NN_WEIGHT_LAYER_FLAG_NORM_SN      0x0100
#define NN_WEIGHT_LAYER_FLAG_NORM_BSSN    0x0200

typedef struct nn_weightLayer_s
{
	nn_layer_t base;

	int flags;

	// weights, bias, output
	//           bs; // batch size
	//           nc; // node count
	//           X;  // dim(bs,1,1,xd)
	nn_tensor_t* X;  // dim(bs,1,1,xd) (reference)
	nn_tensor_t* W;  // dim(nc,1,1,xd)
	nn_tensor_t* B;  // dim(nc,1,1,1)
	nn_tensor_t* Y;  // dim(bs,1,1,nc)

	// Adam - moment estimates
	nn_tensor_t* MW; // dim(nc,1,1,xd)
	nn_tensor_t* VW; // dim(nc,1,1,xd)
	nn_tensor_t* MB; // dim(nc,1,1,1)
	nn_tensor_t* VB; // dim(nc,1,1,1)

	// forward gradients
	// dY_dB; // 1
	// dY_dX; // W : dim(nc,1,1,xd)
	// dY_dW; // X : dim(bs,1,1,xd)

	// backprop gradients
	//           dL_dY; // dim(bs,1,1,nc)
	nn_tensor_t* dL_dW; // dim(nc,1,1,xd)
	nn_tensor_t* dL_dB; // dim(nc,1,1,1)
	nn_tensor_t* dL_dX; // dim(bs,1,1,xd)

	vkk_buffer_t*     sb013_param;
	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1_fp;
	vkk_uniformSet_t* us1_bp;
} nn_weightLayer_t;

nn_weightLayer_t* nn_weightLayer_new(nn_arch_t* arch,
                                     nn_dim_t* dimX,
                                     nn_dim_t* dimW,
                                     int flags);
void              nn_weightLayer_delete(nn_weightLayer_t** _self);
nn_weightLayer_t* nn_weightLayer_import(nn_arch_t* arch,
                                        jsmn_val_t* val);
int               nn_weightLayer_export(nn_weightLayer_t* self,
                                        jsmn_stream_t* stream);

#endif
