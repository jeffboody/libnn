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

#ifndef nn_tensor_H
#define nn_tensor_H

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "../libvkk/vkk.h"
#include "nn_dim.h"

typedef enum
{
	NN_TENSOR_INIT_ZERO   = 0,
	NN_TENSOR_INIT_XAVIER = 1,
	NN_TENSOR_INIT_HE     = 2,
} nn_tensorInit_e;

typedef enum
{
	NN_TENSOR_MODE_IO      = 0,
	NN_TENSOR_MODE_COMPUTE = 1,
} nn_tensorMode_e;

// SN:   Spectral Normalization
// BSSN: Bidirectional Scaled Spectral Normalization
typedef enum
{
	NN_TENSOR_NORM_NONE = 0,
	NN_TENSOR_NORM_SN   = 1,
	NN_TENSOR_NORM_BSSN = 2,
} nn_tensorNorm_e;

#define NN_TENSOR_NORM_COUNT 3

typedef struct nn_tensorOpKUs0Idx_s
{
	uint32_t xn;
	uint32_t yn;
	uint32_t count;
	uint32_t xk;
	uint32_t yk;
	uint32_t depth;
	float    value;
} nn_tensorOpKUs0Idx_t;

typedef struct nn_tensorOpKUs0Data_s
{
	vkk_buffer_t*     sb004_idx;
	vkk_uniformSet_t* us0;
} nn_tensorOpKUs0Data_t;

nn_tensorOpKUs0Data_t* nn_tensorOpKUs0Data_new(nn_tensor_t* X,
                                               nn_tensor_t* Y,
                                               nn_tensorOpKUs0Idx_t* idx);
void                   nn_tensorOpKUs0Data_delete(nn_tensorOpKUs0Data_t** _self);
int                    nn_tensorOpKUs0Data_update(nn_tensorOpKUs0Data_t* self,
                                                  nn_tensor_t* X,
                                                  nn_tensor_t* Y,
                                                  nn_tensorOpKUs0Idx_t* idx);

typedef struct nn_tensor_s
{
	nn_engine_t* engine;

	nn_tensorMode_e mode;

	nn_dim_t dim;

	// IO tensor (optional)
	float* data;

	// compute tensor (optional)
	// sb_dim/sb_data index varies by use case
	vkk_buffer_t*     sb_dim;
	vkk_buffer_t*     sb_data;
	vkk_uniformSet_t* us0;

	// spectral normalization (optional)
	nn_tensorNorm_e   norm;
	vkk_buffer_t*     sb10_data_u1;
	vkk_buffer_t*     sb11_data_v1;
	vkk_buffer_t*     sb12_data_u2;
	vkk_buffer_t*     sb13_data_v2;
	vkk_buffer_t*     sb14_c;
	vkk_uniformSet_t* us1_norm;
} nn_tensor_t;

nn_tensor_t* nn_tensor_new(nn_engine_t* engine,
                           nn_dim_t* dim,
                           nn_tensorInit_e init,
                           nn_tensorMode_e mode);
void         nn_tensor_delete(nn_tensor_t** _self);
int          nn_tensor_import(nn_tensor_t* self,
                              jsmn_val_t* val);
int          nn_tensor_export(nn_tensor_t* self,
                              jsmn_stream_t* stream);
nn_dim_t*    nn_tensor_dim(nn_tensor_t* self);
int          nn_tensor_mode(nn_tensor_t* self);
int          nn_tensor_copy(nn_tensor_t* src,
                            nn_tensor_t* dst,
                            uint32_t src_n,
                            uint32_t dst_n,
                            uint32_t count);
int          nn_tensor_ioClear(nn_tensor_t* self,
                               uint32_t n,
                               uint32_t count);
int          nn_tensor_ioCopy(nn_tensor_t* src,
                              nn_tensor_t* dst,
                              uint32_t src_n,
                              uint32_t dst_n,
                              uint32_t count);
float        nn_tensor_ioGet(nn_tensor_t* self,
                             uint32_t n, uint32_t i,
                             uint32_t j, uint32_t k);
void         nn_tensor_ioSet(nn_tensor_t* self,
                             uint32_t n, uint32_t i,
                             uint32_t j, uint32_t k,
                             float v);
int          nn_tensor_ioExportPng(nn_tensor_t* self,
                                   const char* fname,
                                   uint32_t n,
                                   uint32_t k,
                                   uint32_t depth,
                                   float min,
                                   float max);
int          nn_tensor_computeFill(nn_tensor_t* self,
                                   vkk_hazard_e hazard,
                                   uint32_t n,
                                   uint32_t count,
                                   float value);
int          nn_tensor_computeFillK(nn_tensor_t* self,
                                    vkk_hazard_e hazard,
                                    uint32_t n,
                                    uint32_t count,
                                    uint32_t k,
                                    uint32_t depth,
                                    float value);
int          nn_tensor_computeCopy(nn_tensor_t* src,
                                   nn_tensor_t* dst,
                                   vkk_hazard_e hazard,
                                   uint32_t src_n,
                                   uint32_t dst_n,
                                   uint32_t count);
int          nn_tensor_computeCopyK(nn_tensor_t* src,
                                    nn_tensor_t* dst,
                                    vkk_hazard_e hazard,
                                    uint32_t src_n,
                                    uint32_t dst_n,
                                    uint32_t count,
                                    uint32_t src_k,
                                    uint32_t dst_k,
                                    uint32_t depth);
int          nn_tensor_computeAddK(nn_tensor_t* x,
                                   nn_tensor_t* v,
                                   vkk_hazard_e hazard,
                                   uint32_t xn,
                                   uint32_t vn,
                                   uint32_t count,
                                   uint32_t xk,
                                   uint32_t vk,
                                   uint32_t depth);
int          nn_tensor_computeNormalize(nn_tensor_t* self,
                                        vkk_hazard_e hazard,
                                        nn_tensorNorm_e norm,
                                        float c);
int          nn_tensor_computeStats(nn_tensor_t* self,
                                    vkk_hazard_e hazard,
                                    uint32_t count,
                                    nn_tensorStats_t* stats);

#endif
