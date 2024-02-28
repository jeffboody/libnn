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

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "../libvkk/vkk.h"
#include "nn_layer.h"

typedef enum
{
	NN_FACT_LAYER_FN_ERROR    = -1,
	NN_FACT_LAYER_FN_LINEAR   = 0,
	NN_FACT_LAYER_FN_LOGISTIC = 1,
	NN_FACT_LAYER_FN_RELU     = 2,
	NN_FACT_LAYER_FN_PRELU    = 3,
	NN_FACT_LAYER_FN_TANH     = 4,
	NN_FACT_LAYER_FN_SINK     = 5,
} nn_factLayerFn_e;

#define NN_FACT_LAYER_FN_COUNT 6

typedef struct nn_factLayer_s
{
	nn_layer_t base;

	nn_factLayerFn_e fn;

	// output
	nn_tensor_t* X; // dim(bs,xh,xw,xd) (reference)
	nn_tensor_t* Y; // dim(bs,xh,xw,xd)

	// forward gradients (computed during backprop)
	// dY_dX = dfact(x) : dim(bs,xh,xw,xd)

	// backprop gradients (dL_dY replaced by dL_dX)
	// dL_dY : dim(bs,xh,xw,xd)
	// dL_dX : dim(bs,xh,xw,xd)

	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1;
	vkk_uniformSet_t* us2;
} nn_factLayer_t;

nn_factLayer_t* nn_factLayer_new(nn_arch_t* arch,
                                 nn_dim_t* dimX,
                                 nn_factLayerFn_e fn);
nn_factLayer_t* nn_factLayer_import(nn_arch_t* arch,
                                    jsmn_val_t* val);
int             nn_factLayer_export(nn_factLayer_t* self,
                                    jsmn_stream_t* stream);
void            nn_factLayer_delete(nn_factLayer_t** _self);

#endif
