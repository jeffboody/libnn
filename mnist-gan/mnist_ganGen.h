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

// control the MNIST image range
// TANH:     [-1.0, 1.0]
// LOGISTIC: [ 0.0, 1.0]
#define MNIST_GAN_GEN_TANH
#ifndef MNIST_GAN_GEN_TANH
#define MNIST_GAN_GEN_LOGISTIC
#endif

#define MNIST_GAN_GEN_FC 64

typedef struct
{
	nn_arch_t base;

	// X:  dim(bs, 1, 1, 100)
	// c0: conv_1x1_lrelu
	// c1: convT_4x4_s2_pad_bn_lrelu
	// c2: convT_4x4_s2_pad_bn_lrelu
	// c3: convT_4x4_s2_pad_bn_lrelu
	// c4: conv_3x3_pad_tanh or
	//     conv_3x3_pad_logistic
	nn_coderLayer_t*   c0; // dim(bs,1,1,8*fc*4*4)
	nn_reshapeLayer_t* r1; // dim(bs,4,4,8*fc)
	nn_coderLayer_t*   c1; // dim(bs,8,8,4*fc)
	nn_coderLayer_t*   c2; // dim(bs,16,16,2*fc)
	nn_coderLayer_t*   c3; // dim(bs,32,32,fc)
	nn_coderLayer_t*   c4; // dim(bs,32,32,1)
} mnist_ganGen_t;

mnist_ganGen_t* mnist_ganGen_new(nn_engine_t* engine,
                                 uint32_t bs);
void            mnist_ganGen_delete(mnist_ganGen_t** _self);

#endif
