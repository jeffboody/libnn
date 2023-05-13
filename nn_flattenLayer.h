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

#ifndef nn_flattenLayer_H
#define nn_flattenLayer_H

#include "nn_dim.h"
#include "nn_layer.h"
#include "nn_tensor.h"

typedef struct nn_flattenLayer_s
{
	nn_layer_t base;

	nn_dim_t dimX; // dim(bs,xh,xw,xd)

	// output
	// dim  is flattened
	// data is a reference to X
	nn_tensor_t Y; // dim(bs,1,1,xh*xw*xd)
} nn_flattenLayer_t;

nn_flattenLayer_t* nn_flattenLayer_new(nn_arch_t* arch,
                                       nn_dim_t* dimX);
nn_flattenLayer_t* nn_flattenLayer_import(nn_arch_t* arch,
                                          jsmn_val_t* val);
int                nn_flattenLayer_export(nn_flattenLayer_t* self,
                                          jsmn_stream_t* stream);
void               nn_flattenLayer_delete(nn_flattenLayer_t** _self);

#endif
