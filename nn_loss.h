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

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "nn.h"

typedef nn_tensor_t* (*nn_loss_fn)
                     (nn_loss_t* base, uint32_t bs,
                      nn_tensor_t* Y, nn_tensor_t* Yt);

// loss functions
nn_tensor_t* nn_loss_mse(nn_loss_t* base, uint32_t bs,
                         nn_tensor_t* Y, nn_tensor_t* Yt);

// string/function conversions
const char* nn_loss_string(nn_loss_fn loss_fn);
nn_loss_fn  nn_loss_function(const char* str);

typedef struct nn_loss_s
{
	nn_arch_t* arch;
	nn_loss_fn loss_fn;
	float      loss;

	// backprop gradients
	nn_tensor_t* dL_dY; // dim(bs,yh,yw,yd)
} nn_loss_t;

nn_loss_t* nn_loss_new(nn_arch_t* arch,
                       nn_dim_t* dimY,
                       nn_loss_fn loss_fn);
nn_loss_t* nn_loss_import(nn_arch_t* arch,
                          jsmn_val_t* val);
int        nn_loss_export(nn_loss_t* self,
                          jsmn_stream_t* stream);
void       nn_loss_delete(nn_loss_t** _self);
nn_dim_t*  nn_loss_dimY(nn_loss_t* self);
float      nn_loss_loss(nn_loss_t* self);

#endif
