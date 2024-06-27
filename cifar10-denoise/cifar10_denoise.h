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

#ifndef cifar10_denoise_H
#define cifar10_denoise_H

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

	// optionally select encdec0 or urrdb0
	nn_tensor_t*      Xio;
	nn_tensor_t*      X;
	nn_encdecLayer_t* encdec0;
	nn_urrdbLayer_t*  urrdb0;
	nn_coderLayer_t*  coder1;
	nn_coderLayer_t*  coder2;
	nn_loss_t*        loss;
	nn_tensor_t*      Ytio;
	nn_tensor_t*      Yt;
	nn_tensor_t*      Yio;

	cc_rngNormal_t  rngN;
	cc_rngUniform_t rngU;
} cifar10_denoise_t;

cifar10_denoise_t* cifar10_denoise_new(nn_engine_t* engine,
                                       uint32_t bs,
                                       uint32_t fc,
                                       uint32_t xh,
                                       uint32_t xw,
                                       uint32_t xd,
                                       double mu,
                                       double sigma);
void               cifar10_denoise_delete(cifar10_denoise_t** _self);
cifar10_denoise_t* cifar10_denoise_import(nn_engine_t* engine,
                                          uint32_t xh,
                                          uint32_t xw,
                                          uint32_t xd,
                                          const char* fname);
int                cifar10_denoise_export(cifar10_denoise_t* self,
                                          const char* fname);
int                cifar10_denoise_exportX(cifar10_denoise_t* self,
                                           const char* fname,
                                           uint32_t n);
int                cifar10_denoise_exportYt(cifar10_denoise_t* self,
                                            const char* fname,
                                            uint32_t n);
int                cifar10_denoise_exportY(cifar10_denoise_t* self,
                                           const char* fname,
                                           uint32_t n);
void               cifar10_denoise_sampleXt(cifar10_denoise_t* self,
                                            nn_tensor_t* Xt);
void               cifar10_denoise_sampleXt2(cifar10_denoise_t* self,
                                             nn_tensor_t* Xt,
                                             nn_tensor_t* X,
                                             nn_tensor_t* Yt);
int                cifar10_denoise_train(cifar10_denoise_t* self,
                                         float* _loss);
int                cifar10_denoise_predict(cifar10_denoise_t* self,
                                           uint32_t bs);
uint32_t           cifar10_denoise_bs(cifar10_denoise_t* self);

#endif
