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

#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_dim.h"

/***********************************************************
* public                                                   *
***********************************************************/

int nn_dim_import(nn_dim_t* self, cc_jsmnVal_t* val)
{
	ASSERT(self);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid type=%i", (int) val->type);
		return 0;
	}

	cc_jsmnVal_t* val_count  = NULL;
	cc_jsmnVal_t* val_height = NULL;
	cc_jsmnVal_t* val_width  = NULL;
	cc_jsmnVal_t* val_depth  = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "count") == 0)
			{
				val_count = kv->val;
			}
			else if(strcmp(kv->key, "height") == 0)
			{
				val_height = kv->val;
			}
			else if(strcmp(kv->key, "width") == 0)
			{
				val_width = kv->val;
			}
			else if(strcmp(kv->key, "depth") == 0)
			{
				val_depth = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_count  == NULL) ||
	   (val_height == NULL) ||
	   (val_width  == NULL) ||
	   (val_depth  == NULL))
	{
		LOGE("invalid");
		return 0;
	}

	self->count  = strtol(val_count->data,  NULL, 0);
	self->height = strtol(val_height->data, NULL, 0);
	self->width  = strtol(val_width->data,  NULL, 0);
	self->depth  = strtol(val_depth->data,  NULL, 0);

	return 1;
}

int nn_dim_export(nn_dim_t* self, cc_jsmnStream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= cc_jsmnStream_beginObject(stream);
	ret &= cc_jsmnStream_key(stream, "%s", "count");
	ret &= cc_jsmnStream_int(stream, (int) self->count);
	ret &= cc_jsmnStream_key(stream, "%s", "height");
	ret &= cc_jsmnStream_int(stream, (int) self->height);
	ret &= cc_jsmnStream_key(stream, "%s", "width");
	ret &= cc_jsmnStream_int(stream, (int) self->width);
	ret &= cc_jsmnStream_key(stream, "%s", "depth");
	ret &= cc_jsmnStream_int(stream, (int) self->depth);
	ret &= cc_jsmnStream_end(stream);

	return ret;
}

int nn_dim_validate(nn_dim_t* self,
                    uint32_t n, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);

	if((n >= self->count) || (i >= self->height) ||
	   (j >= self->width) || (k >= self->depth))
	{
		LOGE("n=%u, i=%u, j=%u, k=%u", n, i, j, k);
		LOGE("count=%u, height=%u, width=%u, depth=%u",
		     self->count, self->height,
		     self->width, self->depth);
		return 0;
	}

	return 1;
}

size_t nn_dim_sizeBytes(nn_dim_t* self)
{
	ASSERT(self);

	return sizeof(float)*nn_dim_sizeElements(self);
}

uint32_t nn_dim_sizeElements(nn_dim_t* self)
{
	ASSERT(self);

	return self->count*self->height*self->width*self->depth;
}

int nn_dim_sizeEquals(nn_dim_t* self,
                      nn_dim_t* dim)
{
	ASSERT(self);
	ASSERT(dim);

	if((self->count  != dim->count)  ||
	   (self->height != dim->height) ||
	   (self->width  != dim->width)  ||
	   (self->depth  != dim->depth))
	{
		return 0;
	}

	return 1;
}

size_t nn_dim_strideBytes(nn_dim_t* self)
{
	return sizeof(float)*nn_dim_strideElements(self);
}

uint32_t nn_dim_strideElements(nn_dim_t* self)
{
	ASSERT(self);

	return self->height*self->width*self->depth;
}

int nn_dim_strideEquals(nn_dim_t* self, nn_dim_t* dim)
{
	ASSERT(self);
	ASSERT(dim);

	if((self->height != dim->height) ||
	   (self->width  != dim->width)  ||
	   (self->depth  != dim->depth))
	{
		return 0;
	}

	return 1;
}

void nn_dim_copy(nn_dim_t* src,
                 nn_dim_t* dst)
{
	ASSERT(src);
	ASSERT(dst);

	dst->count  = src->count;
	dst->height = src->height;
	dst->width  = src->width;
	dst->depth  = src->depth;
}
