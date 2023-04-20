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

#include "nn_layer.h"

#define NN_WEIGHT_LAYER_INITMODE_XAVIER 0
#define NN_WEIGHT_LAYER_INITMODE_HE     1

typedef struct nn_weightLayer_s
{
	nn_layer_t base;

	// dimX and dimY will be flattened internally
	// to match the sizes listed below however the
	// external sizes will match the requested sizes

	// weights, bias, output
	//           bs; // batch size
	//           nc; // node count
	//           X;  // dim(bs,1,1,X.d)
	nn_tensor_t* W;  // dim(nc,1,1,X.d)
	nn_tensor_t* B;  // dim(nc,1,1,1)
	nn_tensor_t* Y;  // dim(bs,1,1,nc)

	// forward gradients (batch mean)
	//           dY_dB; // 1
	//           dY_dX; // W        : dim(nc,1,1,X.d)
	nn_tensor_t* dY_dW; // SUM_X/bs : dim(1,1,1,X.d)

	// backprop gradients
	//           dL_dY; // dim(1,1,1,nc) (from next layer)
	nn_tensor_t* dL_dX; // dim(1,1,1,X.d)
} nn_weightLayer_t;

nn_weightLayer_t* nn_weightLayer_new(nn_arch_t* arch,
                                     nn_dim_t* dimX,
                                     nn_dim_t* dimY,
                                     int init_mode);
void              nn_weightLayer_delete(nn_weightLayer_t** _self);

#endif
