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

#ifndef nn_factLayer_H
#define nn_factLayer_H

#include "nn_layer.h"

typedef float (*nn_factLayer_fn)(float x);

// activation functions
float nn_factLayer_linear(float x);
float nn_factLayer_logistic(float x);
float nn_factLayer_ReLU(float x);
float nn_factLayer_PReLU(float x);
float nn_factLayer_tanh(float x);

// activation function derivatives
float nn_factLayer_dlinear(float x);
float nn_factLayer_dlogistic(float x);
float nn_factLayer_dReLU(float x);
float nn_factLayer_dPReLU(float x);
float nn_factLayer_dtanh(float x);

typedef struct nn_factLayer_s
{
	nn_layer_t base;

	// output
	//           X; // dim(bs,X.w,X.h,X.d)
	nn_tensor_t* Y; // dim(X)

	// forward gradients (batch mean)
	nn_tensor_t* dY_dX; // SUM_FACT_X/bs : dim(1,X.w,X.h,X.d)

	// backprop gradients
	//           dL_dY; // dim(1,X.w,X.h,X.d) (from next layer)
	nn_tensor_t* dL_dX; // dim(1,X.w,X.h,X.d)

	// activation functions
	nn_factLayer_fn fact;
	nn_factLayer_fn dfact;
} nn_factLayer_t;

nn_factLayer_t* nn_factLayer_new(nn_dim_t* dim,
                                 nn_factLayer_fn fact,
                                 nn_factLayer_fn dfact);
void            nn_factLayer_delete(nn_factLayer_t** _self);

#endif
