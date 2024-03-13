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
	NN_TENSOR_NORM_MODE_SN   = 0,
	NN_TENSOR_NORM_MODE_BSSN = 1,
} nn_tensorNormMode_e;

typedef struct nn_tensor_s
{
	nn_engine_t* engine;

	nn_tensorMode_e tensor_mode;

	nn_dim_t dim;

	// IO tensor (optional)
	float* data;

	// compute tensor (optional)
	// sb_dim/sb_data index varies by layer but are
	// sb00_dim/sb01_data for tensor compute shaders
	vkk_buffer_t*     sb_dim;
	vkk_buffer_t*     sb_data;
	vkk_uniformSet_t* us0;

	// spectral normalization (optional)
	vkk_buffer_t*     sb20_data_u1;
	vkk_buffer_t*     sb21_data_v1;
	vkk_buffer_t*     sb22_data_u2;
	vkk_buffer_t*     sb23_data_v2;
	vkk_buffer_t*     sb24_c;
	vkk_uniformSet_t* us2;
} nn_tensor_t;

nn_tensor_t* nn_tensor_new(nn_engine_t* engine,
                           nn_dim_t* dim,
                           nn_tensorInit_e init,
                           nn_tensorMode_e tensor_mode);
void         nn_tensor_delete(nn_tensor_t** _self);
void         nn_tensor_print(nn_tensor_t* self,
                             const char* name);
int          nn_tensor_exportPng(nn_tensor_t* self,
                                 const char* fname,
                                 uint32_t n,
                                 uint32_t k0,
                                 uint32_t k1,
                                 float min,
                                 float max);
int          nn_tensor_load(nn_tensor_t* self,
                            jsmn_val_t* val);
int          nn_tensor_store(nn_tensor_t* self,
                             jsmn_stream_t* stream);
int          nn_tensor_clear(nn_tensor_t* self,
                             vkk_hazard_e hazard);
int          nn_tensor_normalize(nn_tensor_t* self,
                                 vkk_hazard_e hazard,
                                 nn_tensorNormMode_e norm,
                                 float c);
int          nn_tensor_computeStats(nn_tensor_t* self,
                                    uint32_t count,
                                    vkk_hazard_e hazard,
                                    nn_tensorStats_t* stats);
float        nn_tensor_get(nn_tensor_t* self,
                           uint32_t n, uint32_t i,
                           uint32_t j, uint32_t k);
void         nn_tensor_set(nn_tensor_t* self,
                           uint32_t n, uint32_t i,
                           uint32_t j, uint32_t k,
                           float v);
nn_dim_t*    nn_tensor_dim(nn_tensor_t* self);
int          nn_tensor_blit(nn_tensor_t* src,
                            nn_tensor_t* dst,
                            uint32_t count,
                            uint32_t src_offset,
                            uint32_t dst_offset);

#endif
