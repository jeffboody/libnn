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

#ifndef nn_batchNormLayer_H
#define nn_batchNormLayer_H

#include "nn_layer.h"

typedef struct nn_batchNormLayer_s
{
	nn_layer_t base;

	// gamma, beta, output
	nn_tensor_t* G; // dim(1,1,1,xd)
	nn_tensor_t* B; // dim(1,1,1,xd)
	nn_tensor_t* Y; // dim(bs,xh,xw,xd)

	// mini-batch mean/variance
	nn_tensor_t* Xmean_mb; // dim(1,1,1,xd)
	nn_tensor_t* Xvar_mb;  // dim(1,1,1,xd)

	// running averages
	nn_tensor_t* Xmean_ra; // dim(1,1,1,xd)
	nn_tensor_t* Xvar_ra;  // dim(1,1,1,xd)

	// forward gradients
	nn_tensor_t* dXvar_dX;    // dim(1,1,1,xd)
	nn_tensor_t* dXhat_dX;    // dim(1,1,1,xd)
	nn_tensor_t* dXhat_dXvar; // dim(1,xh,xw,xd)
	nn_tensor_t* dY_dG;       // dim(1,xh,xw,xd)

	// backprop gradients
	//           dL_dY;     // dim(1,xh,xw,xd)
	nn_tensor_t* dL_dX;     // dim(1,xh,xw,xd)
	nn_tensor_t* dL_dXvar;  // dim(1,1,1,xd)
	nn_tensor_t* dL_dXmean; // dim(1,1,1,xd)
} nn_batchNormLayer_t;

nn_batchNormLayer_t* nn_batchNormLayer_new(nn_arch_t* arch,
                                           nn_dim_t* dim);
void                 nn_batchNormLayer_delete(nn_batchNormLayer_t** _self);

#endif
