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

#ifndef cifar10_upsample_H
#define cifar10_upsample_H

#include "libcc/rng/cc_rngNormal.h"
#include "libcc/rng/cc_rngUniform.h"
#include "libnn/nn_arch.h"
#include "libnn/nn.h"
#include "libvkk/vkk_platform.h"
#include "cifar10_lanczos.h"

typedef struct
{
	nn_arch_t base;

	uint32_t bs;
	uint32_t fc;

	nn_tensor_t*       Xio;
	nn_tensor_t*       X;
	nn_urrdbLayer_t*   urrdb0;
	nn_coderLayer_t*   coder1;
	nn_lanczosLayer_t* up1;
	nn_coderLayer_t*   coder2;
	nn_lanczosLayer_t* down2;
	nn_loss_t*         loss;
	nn_tensor_t*       Ytio;
	nn_tensor_t*       Yt;
	nn_tensor_t*       Yio;
	nn_tensor_t*       Uio;

	cifar10_lanczos_t* lanczos;

	cc_rngUniform_t rngU;
} cifar10_upsample_t;

cifar10_upsample_t* cifar10_upsample_new(nn_engine_t* engine,
                                         uint32_t bs,
                                         uint32_t fc,
                                         uint32_t xh,
                                         uint32_t xw,
                                         uint32_t xd);
void                cifar10_upsample_delete(cifar10_upsample_t** _self);
cifar10_upsample_t* cifar10_upsample_import(nn_engine_t* engine,
                                            uint32_t xh,
                                            uint32_t xw,
                                            uint32_t xd,
                                            const char* fname);
int                 cifar10_upsample_export(cifar10_upsample_t* self,
                                            const char* fname);
int                 cifar10_upsample_exportX(cifar10_upsample_t* self,
                                             const char* fname,
                                             uint32_t n);
int                 cifar10_upsample_exportYt(cifar10_upsample_t* self,
                                              const char* fname,
                                              uint32_t n);
int                 cifar10_upsample_exportYLR(cifar10_upsample_t* self,
                                               const char* fname,
                                               uint32_t n);
int                 cifar10_upsample_exportYHR(cifar10_upsample_t* self,
                                               const char* fname,
                                               uint32_t n);
int                 cifar10_upsample_exportLLY(cifar10_upsample_t* self,
                                               const char* fname,
                                               uint32_t n);
int                 cifar10_upsample_exportLRY(cifar10_upsample_t* self,
                                               const char* fname,
                                               uint32_t n);
void                cifar10_upsample_sampleXt(cifar10_upsample_t* self,
                                              nn_tensor_t* Xt);
void                cifar10_upsample_sampleXt2(cifar10_upsample_t* self,
                                               nn_tensor_t* Xt,
                                               nn_tensor_t* X,
                                               nn_tensor_t* Yt);
int                 cifar10_upsample_train(cifar10_upsample_t* self,
                                           float* _loss);
int                 cifar10_upsample_predict(cifar10_upsample_t* self,
                                             uint32_t bs);
uint32_t            cifar10_upsample_bs(cifar10_upsample_t* self);

#endif
