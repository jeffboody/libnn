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

#ifndef nn_loss_H
#define nn_loss_H

#include "nn.h"

typedef nn_tensor_t* (*nn_loss_backpropFn)
                     (nn_loss_t* base, uint32_t bs,
                      nn_tensor_t* Y, nn_tensor_t* Yt);
typedef nn_dim_t* (*nn_loss_dimFn)
                  (nn_loss_t* base);

typedef struct nn_lossInfo_s
{
	nn_arch_t*         arch;
	nn_loss_backpropFn backprop_fn;
	nn_loss_dimFn      dimY_fn;
} nn_lossInfo_t;

typedef struct nn_loss_s
{
	nn_arch_t*         arch;
	nn_loss_backpropFn backprop_fn;
	nn_loss_dimFn      dimY_fn;
	float              loss;
} nn_loss_t;

nn_loss_t* nn_loss_new(size_t base_size,
                       nn_lossInfo_t* info);
void       nn_loss_delete(nn_loss_t** _self);
nn_dim_t*  nn_loss_dimY(nn_loss_t* self);

#endif
