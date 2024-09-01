/*
 * Copyright (c) 2024 Jeff Boody
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

#ifndef mnist_ganGen_H
#define mnist_ganGen_H

#include "../../libnn/nn_arch.h"
#include "../../libnn/nn.h"

typedef struct
{
	nn_arch_t base;

	// X: dim(bs, 1, 1, 100)
	nn_weightLayer_t*  w0; // dim(bs,1,1,6272) (7x7x128)
	nn_factLayer_t*    f0; // dim(bs,1,1,6272) (7x7x128)
	nn_reshapeLayer_t* r1; // dim(bs,7,7,128)
	nn_coderLayer_t*   c2; // dim(bs,14,14,128) (convT_4x4_s2_lrelu)
	nn_coderLayer_t*   c3; // dim(bs,28,28,128) (convT_4x4_s2_lrelu)
	nn_coderLayer_t*   c4; // dim(bs,28,28,1)   (conv_7x7_sigmoid)
} mnist_ganGen_t;

mnist_ganGen_t* mnist_ganGen_new(nn_engine_t* engine,
                                 uint32_t bs);
void            mnist_ganGen_delete(mnist_ganGen_t** _self);

#endif
