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

// control the discriminator activation and loss functions
// CLASSIC: logistic and bce
// LSGAN:   linear   and mse
#define MNIST_GAN_DISC_CLASSIC
#ifndef MNIST_GAN_DISC_CLASSIC
#define MNIST_GAN_DISC_LSGAN
#endif

#define MNIST_GAN_DISC_FC 64

typedef struct
{
	nn_arch_t base;

	// X: dim(bs, 32, 32, 1)
	// c0: conv4x4_s2_pad_lrelu
	// c1: conv4x4_s2_pad_bn_lrelu
	// c2: conv4x4_s2_pad_bn_lrelu
	// c3: conv4x4_s2_pad_bn_lrelu
	// c4: conv1x1_pad_nobias_logistic or (classic)
	//     conv1x1_pad_nobias_linear      (LSGAN)
	nn_coderLayer_t*   c0; // dim(bs,16,16,fc)
	nn_coderLayer_t*   c1; // dim(bs,8,8,2*fc)
	nn_coderLayer_t*   c2; // dim(bs,4,4,4*fc)
	nn_coderLayer_t*   c3; // dim(bs,2,2,8*fc)
	nn_reshapeLayer_t* r3; // dim(bs,1,1,8*fc*2*2)
	nn_coderLayer_t*   c4; // dim(bs,1,1,1)
} mnist_ganDisc_t;

mnist_ganDisc_t* mnist_ganDisc_new(nn_engine_t* engine,
                                   uint32_t bs);
void             mnist_ganDisc_delete(mnist_ganDisc_t** _self);

#endif
