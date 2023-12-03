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

#ifndef cifar10_disc_H
#define cifar10_disc_H

#include "libnn/cifar10-denoise/cifar10_denoise.h"
#include "libnn/nn_arch.h"
#include "libnn/nn.h"
#include "libvkk/vkk_platform.h"

typedef struct
{
	nn_arch_t base;

	uint32_t bs;
	uint32_t fc;

	nn_tensor_t*         X;
	nn_tensor_t*         dL_dY;
	nn_batchNormLayer_t* bn0;
	nn_coderLayer_t*     coder1;
	nn_coderLayer_t*     coder2;
	nn_coderLayer_t*     coder3;
	nn_convLayer_t*      convO;
	nn_factLayer_t*      factO;
	nn_loss_t*           loss;
	nn_tensor_t*         Yt;
	nn_tensor_t*         Y;
} cifar10_disc_t;

cifar10_disc_t* cifar10_disc_new(nn_engine_t* engine,
                                 uint32_t bs,
                                 uint32_t fc,
                                 uint32_t xh,
                                 uint32_t xw,
                                 uint32_t xd);
void            cifar10_disc_delete(cifar10_disc_t** _self);
cifar10_disc_t* cifar10_disc_import(nn_engine_t* engine,
                                    uint32_t xh,
                                    uint32_t xw,
                                    uint32_t xd,
                                    const char* fname);
int             cifar10_disc_export(cifar10_disc_t* self,
                                    const char* fname);
int             cifar10_disc_exportXd0(cifar10_disc_t* self,
                                       const char* fname,
                                       uint32_t n);
int             cifar10_disc_exportXd1(cifar10_disc_t* self,
                                       const char* fname,
                                       uint32_t n);
int             cifar10_disc_export_dL_dY0(cifar10_disc_t* self,
                                           const char* fname,
                                           uint32_t n);
int             cifar10_disc_export_dL_dY1(cifar10_disc_t* self,
                                           const char* fname,
                                           uint32_t n);
int             cifar10_disc_exportY(cifar10_disc_t* self,
                                     const char* fname,
                                     uint32_t n);
void            cifar10_disc_sampleXt(cifar10_disc_t* self,
                                      cifar10_denoise_t* dn,
                                      nn_tensor_t* Xt);
int             cifar10_disc_train(cifar10_disc_t* self,
                                   float* _loss);
int             cifar10_disc_predict(cifar10_disc_t* self,
                                     uint32_t bs);
uint32_t        cifar10_disc_bs(cifar10_disc_t* self);

#endif
