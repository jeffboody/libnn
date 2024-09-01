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

#ifndef mnist_ganDisc_H
#define mnist_ganDisc_H

#include "../../libnn/nn_arch.h"
#include "../../libnn/nn.h"

typedef struct
{
	nn_arch_t base;

	// X: dim(bs, 28, 28, 1)
	nn_coderLayer_t*   c0; // dim(bs,14,14,64) (conv3x3_s2_lrelu)
	nn_coderLayer_t*   c1; // dim(bs,7,7,64)   (conv3x3_s2_bn_lrelu)
	nn_reshapeLayer_t* r3; // dim(bs,1,1,3136) (7x7x64)
	nn_weightLayer_t*  w4; // dim(bs,1,1,1)
	nn_factLayer_t*    o5; // dim(bs,1,1,1) (linear)
} mnist_ganDisc_t;

mnist_ganDisc_t* mnist_ganDisc_new(nn_engine_t* engine,
                                   uint32_t bs);
void             mnist_ganDisc_delete(mnist_ganDisc_t** _self);

#endif
