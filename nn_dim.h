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

#ifndef nn_dim_H
#define nn_dim_H

#include "../libcc/jsmn/cc_jsmnStream.h"
#include "../libcc/jsmn/cc_jsmnWrapper.h"
#include "nn.h"

typedef struct nn_dim_s
{
	uint32_t count;
	uint32_t height;
	uint32_t width;
	uint32_t depth;
} nn_dim_t;

int      nn_dim_import(nn_dim_t* self,
                       cc_jsmnVal_t* val);
int      nn_dim_export(nn_dim_t* self,
                      cc_jsmnStream_t* stream);
int      nn_dim_validate(nn_dim_t* self,
                         uint32_t n, uint32_t i,
                         uint32_t j, uint32_t k);
size_t   nn_dim_sizeBytes(nn_dim_t* self);
uint32_t nn_dim_sizeElements(nn_dim_t* self);
int      nn_dim_sizeEquals(nn_dim_t* self,
                           nn_dim_t* dim);
size_t   nn_dim_strideBytes(nn_dim_t* self);
uint32_t nn_dim_strideElements(nn_dim_t* self);
int      nn_dim_strideEquals(nn_dim_t* self,
                             nn_dim_t* dim);
void     nn_dim_copy(nn_dim_t* src,
                     nn_dim_t* dst);

#endif
