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

// string/function conversions
const char*     nn_factLayer_string(nn_factLayer_fn fact_fn);
nn_factLayer_fn nn_factLayer_function(const char* str);

typedef struct nn_factLayer_s
{
	nn_layer_t base;

	// output
	nn_tensor_t* X; // dim(bs,xh,xw,xd) (reference)
	nn_tensor_t* Y; // dim(bs,xh,xw,xd)

	// forward gradients (computed during backprop)
	// dY_dX = dfact(x) : dim(bs,xh,xw,xd)

	// backprop gradients (dL_dY replaced by dL_dX)
	// dL_dY : dim(bs,xh,xw,xd)
	// dL_dX : dim(bs,xh,xw,xd)

	// activation functions
	nn_factLayer_fn fact_fn;
	nn_factLayer_fn dfact_fn;
} nn_factLayer_t;

nn_factLayer_t* nn_factLayer_new(nn_arch_t* arch,
                                 nn_dim_t* dimX,
                                 nn_factLayer_fn fact_fn,
                                 nn_factLayer_fn dfact_fn);
nn_factLayer_t* nn_factLayer_import(nn_arch_t* arch,
                                    jsmn_val_t* val);
int             nn_factLayer_export(nn_factLayer_t* self,
                                    jsmn_stream_t* stream);
void            nn_factLayer_delete(nn_factLayer_t** _self);

#endif
