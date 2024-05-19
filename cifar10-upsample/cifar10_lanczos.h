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

#ifndef cifar10_lanczos_H
#define cifar10_lanczos_H

#include "libnn/nn_arch.h"
#include "libnn/nn.h"

typedef struct
{
	nn_arch_t base;

	int X_dirty;
	int LT_dirty;
	int LY_dirty;
	int RY_dirty;

	nn_tensor_t* Xio;
	nn_tensor_t* LTio;
	nn_tensor_t* LYio;
	nn_tensor_t* RYio;
	nn_tensor_t* X; // reference
	nn_tensor_t* LT; // reference
	nn_tensor_t* LY; // reference

	nn_lanczosLayer_t*     lanczosL;
	nn_lanczosResampler_t* lanczosR;
} cifar10_lanczos_t;

cifar10_lanczos_t* cifar10_lanczos_new(nn_engine_t* engine,
                                       nn_dim_t* dimX,
                                       nn_dim_t* dimY);
void               cifar10_lanczos_delete(cifar10_lanczos_t** _self);
nn_tensor_t*       cifar10_lanczos_computeFp(cifar10_lanczos_t* self,
                                             int flags,
                                             uint32_t bs,
                                             nn_tensor_t* X);
int                cifar10_lanczos_exportX(cifar10_lanczos_t* self,
                                           const char* fname,
                                           uint32_t n);
int                cifar10_lanczos_exportLT(cifar10_lanczos_t* self,
                                            const char* fname,
                                            uint32_t n);
int                cifar10_lanczos_exportLY(cifar10_lanczos_t* self,
                                            const char* fname,
                                            uint32_t n);
int                cifar10_lanczos_exportRY(cifar10_lanczos_t* self,
                                            const char* fname,
                                            uint32_t n);

#endif
