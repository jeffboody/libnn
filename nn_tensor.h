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

typedef struct nn_tensorOpUs0Idx_s
{
	uint32_t x1n;
	uint32_t x2n;
	uint32_t yn;
	uint32_t count;
	uint32_t x1i;
	uint32_t x2i;
	uint32_t yi;
	uint32_t height;
	uint32_t x1j;
	uint32_t x2j;
	uint32_t yj;
	uint32_t width;
	uint32_t x1k;
	uint32_t x2k;
	uint32_t yk;
	uint32_t depth;
	float    value;
} nn_tensorOpUs0Idx_t;

typedef struct nn_tensorOpUs0Data_s
{
	vkk_buffer_t*     sb006_idx;
	vkk_uniformSet_t* us0;
} nn_tensorOpUs0Data_t;

nn_tensorOpUs0Data_t* nn_tensorOpUs0Data_new(nn_tensor_t* X1,
                                             nn_tensor_t* X2,
                                             nn_tensor_t* Y,
                                             nn_tensorOpUs0Idx_t* idx);
void                  nn_tensorOpUs0Data_delete(nn_tensorOpUs0Data_t** _self);
int                   nn_tensorOpUs0Data_update(nn_tensorOpUs0Data_t* self,
                                                nn_tensor_t* X1,
                                                nn_tensor_t* X2,
                                                nn_tensor_t* Y,
                                                nn_tensorOpUs0Idx_t* idx);

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
	vkk_buffer_t*     sb100_data_u1;
	vkk_buffer_t*     sb101_data_v1;
	vkk_buffer_t*     sb102_data_u2;
	vkk_buffer_t*     sb103_data_v2;
	vkk_buffer_t*     sb104_c;
	vkk_uniformSet_t* us1_norm;
} nn_tensor_t;

/*
 * Tensors may be stored in CPU accessible memory or GPU
 * only memory. The engine, arch, layers and loss will only
 * accept compute tensors unless specified otherwise. Each
 * tensor function has different memory requirements in
 * order operate. The IO and compute functions require the
 * corresponding mode to be set. The remaining functions may
 * be used in either case. It is the users responsibility to
 * copy IO tensors to/from compute tensors as required.
 *
 * The compute tensor functions may only be called when the
 * engine is in the computing state. The auxillary functions
 * (e.g. import/export/copy) may be called on a compute
 * tensor which is not in use by the compute engine. Keep in
 * mind that that the compute pipeline operations are not
 * guaranteed to complete until the engine compute pass
 * ends.
 *
 * The compute functions submit their commands to be
 * executed on the GPU in an arbitrary order. In some
 * scenarios, a set of commands may require a specific
 * exection order to produce the correct result. The correct
 * execution order is guaranteed by specifying a hazard flag
 * which describes write-after-read (WAR) and
 * read-after-write (RAW) conflicts. The computeOp
 * functions may be used to write to separate regions of a
 * tensor across multiple calls, however, this should be
 * treated as a RAW conflict.
 */
nn_tensor_t*    nn_tensor_new(nn_engine_t* engine,
                              nn_dim_t* dim,
                              nn_tensorInit_e init,
                              nn_tensorMode_e mode);
void            nn_tensor_delete(nn_tensor_t** _self);
int             nn_tensor_import(nn_tensor_t* self,
                                 jsmn_val_t* val);
int             nn_tensor_export(nn_tensor_t* self,
                                 jsmn_stream_t* stream);
nn_dim_t*       nn_tensor_dim(nn_tensor_t* self);
nn_tensorMode_e nn_tensor_mode(nn_tensor_t* self);
int             nn_tensor_copy(nn_tensor_t* X,
                               nn_tensor_t* Y,
                               uint32_t xn,
                               uint32_t yn,
                               uint32_t count);
int             nn_tensor_ioClear(nn_tensor_t* self,
                                  uint32_t n,
                                  uint32_t count);
int             nn_tensor_ioCopy(nn_tensor_t* X,
                                 nn_tensor_t* Y,
                                 uint32_t xn,
                                 uint32_t yn,
                                 uint32_t count);
float           nn_tensor_ioGet(nn_tensor_t* self,
                                uint32_t n, uint32_t i,
                                uint32_t j, uint32_t k);
void            nn_tensor_ioSet(nn_tensor_t* self,
                                uint32_t n, uint32_t i,
                                uint32_t j, uint32_t k,
                                float v);
int             nn_tensor_ioExportPng(nn_tensor_t* self,
                                      const char* fname,
                                      uint32_t n,
                                      uint32_t k,
                                      uint32_t depth,
                                      float min,
                                      float max);
int             nn_tensor_computeFill(nn_tensor_t* self,
                                      vkk_hazard_e hazard,
                                      uint32_t n,
                                      uint32_t count,
                                      float value);
int             nn_tensor_computeCopy(nn_tensor_t* X,
                                      nn_tensor_t* Y,
                                      vkk_hazard_e hazard,
                                      uint32_t xn,
                                      uint32_t yn,
                                      uint32_t count);
int             nn_tensor_computeFillOp(nn_tensor_t* self,
                                        vkk_hazard_e hazard,
                                        uint32_t n,
                                        uint32_t count,
                                        uint32_t i,
                                        uint32_t height,
                                        uint32_t j,
                                        uint32_t width,
                                        uint32_t k,
                                        uint32_t depth,
                                        float value);
int             nn_tensor_computeCopyOp(nn_tensor_t* X,
                                        nn_tensor_t* Y,
                                        vkk_hazard_e hazard,
                                        uint32_t xn,
                                        uint32_t yn,
                                        uint32_t count,
                                        uint32_t xi,
                                        uint32_t yi,
                                        uint32_t height,
                                        uint32_t xj,
                                        uint32_t yj,
                                        uint32_t width,
                                        uint32_t xk,
                                        uint32_t yk,
                                        uint32_t depth);
int             nn_tensor_computeAddOp(nn_tensor_t* X1,
                                       nn_tensor_t* X2,
                                       nn_tensor_t* Y,
                                       vkk_hazard_e hazard,
                                       uint32_t x1n,
                                       uint32_t x2n,
                                       uint32_t yn,
                                       uint32_t count,
                                       uint32_t x1i,
                                       uint32_t x2i,
                                       uint32_t yi,
                                       uint32_t height,
                                       uint32_t x1j,
                                       uint32_t x2j,
                                       uint32_t yj,
                                       uint32_t width,
                                       uint32_t x1k,
                                       uint32_t x2k,
                                       uint32_t yk,
                                       uint32_t depth);
int             nn_tensor_computeMixOp(nn_tensor_t* X1,
                                       nn_tensor_t* X2,
                                       nn_tensor_t* Y,
                                       vkk_hazard_e hazard,
                                       uint32_t x1n,
                                       uint32_t x2n,
                                       uint32_t yn,
                                       uint32_t count,
                                       uint32_t x1i,
                                       uint32_t x2i,
                                       uint32_t yi,
                                       uint32_t height,
                                       uint32_t x1j,
                                       uint32_t x2j,
                                       uint32_t yj,
                                       uint32_t width,
                                       uint32_t x1k,
                                       uint32_t x2k,
                                       uint32_t yk,
                                       uint32_t depth,
                                       float value);
int             nn_tensor_computeScaleOp(nn_tensor_t* X,
                                         nn_tensor_t* Y,
                                         vkk_hazard_e hazard,
                                         uint32_t xn,
                                         uint32_t yn,
                                         uint32_t count,
                                         uint32_t xi,
                                         uint32_t yi,
                                         uint32_t height,
                                         uint32_t xj,
                                         uint32_t yj,
                                         uint32_t width,
                                         uint32_t xk,
                                         uint32_t yk,
                                         uint32_t depth,
                                         float value);
int             nn_tensor_computeScaleAddOp(nn_tensor_t* X1,
                                            nn_tensor_t* X2,
                                            nn_tensor_t* Y,
                                            vkk_hazard_e hazard,
                                            uint32_t x1n,
                                            uint32_t x2n,
                                            uint32_t yn,
                                            uint32_t count,
                                            uint32_t x1i,
                                            uint32_t x2i,
                                            uint32_t yi,
                                            uint32_t height,
                                            uint32_t x1j,
                                            uint32_t x2j,
                                            uint32_t yj,
                                            uint32_t width,
                                            uint32_t x1k,
                                            uint32_t x2k,
                                            uint32_t yk,
                                            uint32_t depth,
                                            float value);
int             nn_tensor_computeNormalize(nn_tensor_t* self,
                                           vkk_hazard_e hazard,
                                           nn_tensorNorm_e norm,
                                           float c);
int             nn_tensor_computeStats(nn_tensor_t* self,
                                       vkk_hazard_e hazard,
                                       uint32_t count,
                                       nn_tensorStats_t* stats);

#endif
