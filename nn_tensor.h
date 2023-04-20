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

#ifndef nn_tensor_H
#define nn_tensor_H

#include "nn_dim.h"

typedef struct nn_tensor_s
{
	nn_dim_t dim;
	float*   data;
} nn_tensor_t;

nn_tensor_t* nn_tensor_new(nn_dim_t* dim);
void         nn_tensor_delete(nn_tensor_t** _self);
void         nn_tensor_flatten(nn_tensor_t* self,
                               nn_tensor_t* flat);
void         nn_tensor_clear(nn_tensor_t* self);
float        nn_tensor_get(nn_tensor_t* self,
                           uint32_t i, uint32_t x,
                           uint32_t y, uint32_t z);
void         nn_tensor_set(nn_tensor_t* self,
                           uint32_t i, uint32_t x,
                           uint32_t y, uint32_t z,
                           float val);
void         nn_tensor_add(nn_tensor_t* self,
                           uint32_t i, uint32_t x,
                           uint32_t y, uint32_t z,
                           float val);
void         nn_tensor_mul(nn_tensor_t* self,
                           uint32_t i, uint32_t x,
                           uint32_t y, uint32_t z,
                           float val);
nn_dim_t*    nn_tensor_dim(nn_tensor_t* self);
int          nn_tensor_blit(nn_tensor_t* src,
                            nn_tensor_t* dst,
                            uint32_t srci, uint32_t dsti);

#endif