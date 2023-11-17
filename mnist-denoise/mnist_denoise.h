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

#ifndef mnist_denoise_H
#define mnist_denoise_H

#include "libcc/rng/cc_rngNormal.h"
#include "libcc/rng/cc_rngUniform.h"
#include "libnn/nn_arch.h"
#include "libnn/nn.h"
#include "libvkk/vkk_platform.h"

typedef struct
{
	nn_arch_t base;

	uint32_t bs;
	uint32_t fc;

	double mu;
	double sigma;

	nn_tensor_t*         X;
	nn_batchNormLayer_t* bn0;
	nn_coderLayer_t*     enc1;
	nn_coderLayer_t*     enc2;
	nn_coderLayer_t*     dec3;
	nn_coderLayer_t*     dec4;
	nn_convLayer_t*      convO;
	nn_factLayer_t*      factO;
	nn_loss_t*           loss;
	nn_tensor_t*         Yt;
	nn_tensor_t*         Y;

	cc_rngNormal_t  rngN;
	cc_rngUniform_t rngU;
} mnist_denoise_t;

mnist_denoise_t* mnist_denoise_new(nn_engine_t* engine,
                                   uint32_t bs,
                                   uint32_t fc,
                                   uint32_t xh,
                                   uint32_t xw,
                                   double mu,
                                   double sigma);
void             mnist_denoise_delete(mnist_denoise_t** _self);
mnist_denoise_t* mnist_denoise_import(nn_engine_t* engine,
                                      uint32_t xh,
                                      uint32_t xw,
                                      const char* fname);
int              mnist_denoise_export(mnist_denoise_t* self,
                                      const char* fname);
int              mnist_denoise_exportX(mnist_denoise_t* self,
                                       const char* fname,
                                       uint32_t n);
int              mnist_denoise_exportYt(mnist_denoise_t* self,
                                        const char* fname,
                                        uint32_t n);
int              mnist_denoise_exportY(mnist_denoise_t* self,
                                       const char* fname,
                                       uint32_t n);
void             mnist_denoise_sampleXt(mnist_denoise_t* self,
                                        nn_tensor_t* Xt);
void             mnist_denoise_sampleXt2(mnist_denoise_t* self,
                                         nn_tensor_t* Xt,
                                         nn_tensor_t* X,
                                         nn_tensor_t* Yt);
int              mnist_denoise_train(mnist_denoise_t* self,
                                     float* _loss);
int              mnist_denoise_predict(mnist_denoise_t* self,
                                       uint32_t bs);
uint32_t         mnist_denoise_bs(mnist_denoise_t* self);

#endif
