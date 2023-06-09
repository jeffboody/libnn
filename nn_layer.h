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

#ifndef nn_layer_H
#define nn_layer_H

#include "nn.h"

#define NN_LAYER_MODE_TRAIN   0
#define NN_LAYER_MODE_PREDICT 1

typedef nn_tensor_t* (*nn_layer_forwardPassFn)
                     (nn_layer_t* base, int mode,
                      uint32_t bs, nn_tensor_t* X);
typedef nn_tensor_t* (*nn_layer_backpropFn)
                     (nn_layer_t* base, uint32_t bs,
                      nn_tensor_t* dL_dY);
typedef nn_dim_t* (*nn_layer_dimFn)
                  (nn_layer_t* base);

typedef struct nn_layerInfo_s
{
	nn_arch_t*             arch;
	nn_layer_forwardPassFn forward_pass_fn;
	nn_layer_backpropFn    backprop_fn;
	nn_layer_dimFn         dimX_fn;
	nn_layer_dimFn         dimY_fn;
} nn_layerInfo_t;

typedef struct nn_layer_s
{
	nn_arch_t*             arch;
	nn_layer_forwardPassFn forward_pass_fn;
	nn_layer_backpropFn    backprop_fn;
	nn_layer_dimFn         dimX_fn;
	nn_layer_dimFn         dimY_fn;
} nn_layer_t;

nn_layer_t*  nn_layer_new(size_t base_size,
                          nn_layerInfo_t* info);
void         nn_layer_delete(nn_layer_t** _self);
nn_tensor_t* nn_layer_forwardPass(nn_layer_t* self,
                                  int mode, uint32_t bs,
                                  nn_tensor_t* X);
nn_tensor_t* nn_layer_backprop(nn_layer_t* self,
                               uint32_t bs,
                               nn_tensor_t* dL_dY);
nn_dim_t*    nn_layer_dimX(nn_layer_t* self);
nn_dim_t*    nn_layer_dimY(nn_layer_t* self);

#endif
