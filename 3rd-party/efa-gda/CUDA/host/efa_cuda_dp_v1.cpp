// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#include "efa_cuda_dp_types.h"
#include "efa_cuda_dp_v1.h"

static const char *efa_cuda_error_string_v1(CUresult result)
{
	const char *error_string = nullptr;

	if (cuGetErrorString(result, &error_string) != CUDA_SUCCESS || !error_string)
		return "unknown CUDA error";

	return error_string;
}

static bool efa_cuda_attrs_extension_is_cleared_v1(const void *attrs, size_t attrs_size,
						    uint32_t inlen)
{
	const uint8_t *extension;
	size_t extension_size;

	if (inlen <= attrs_size)
		return true;

	extension = static_cast<const uint8_t *>(attrs) + attrs_size;
	extension_size = inlen - attrs_size;
	for (size_t i = 0; i < extension_size; i++) {
		if (extension[i])
			return false;
	}

	return true;
}

static void efa_cuda_init_wq_v1(efa_cuda_wq_v1 *wq, uint8_t *buffer,
				uint32_t *doorbell, uint32_t num_entries,
				uint32_t max_batch, int phase)
{
	wq->buf = buffer;
	wq->db = doorbell;
	wq->max_wqes = num_entries;
	wq->max_batch = max_batch;
	wq->queue_mask = num_entries - 1;
	wq->queue_size_shift = __builtin_ctz(num_entries);
	wq->phase = phase;
}

template <typename Object, typename Attrs>
static Object *efa_cuda_create_device_object_v1(Attrs *attrs, uint32_t inlen,
						 int (*init)(Object *, Attrs *, uint32_t),
						 const char *object_name)
{
	Object host_object;
	CUdeviceptr device_memory;
	CUresult cuda_result;
	int ret;

	ret = init(&host_object, attrs, inlen);
	if (ret)
		return nullptr;

	cuda_result = cuMemAlloc(&device_memory, sizeof(host_object));
	if (cuda_result != CUDA_SUCCESS) {
		printf("Failed to allocate device memory for %s: %s\n", object_name,
		       efa_cuda_error_string_v1(cuda_result));
		return nullptr;
	}

	cuda_result = cuMemcpyHtoD(device_memory, &host_object, sizeof(host_object));
	if (cuda_result != CUDA_SUCCESS) {
		cuMemFree(device_memory);
		printf("Failed to copy %s to device: %s\n", object_name,
		       efa_cuda_error_string_v1(cuda_result));
		return nullptr;
	}

	return reinterpret_cast<Object *>(device_memory);
}

static int efa_cuda_init_cq_v1(efa_cuda_cq_v1 *cq, efa_cuda_cq_attrs_v1 *attrs,
			       uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared_v1(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->num_entries) != 1) {
		printf("CQ size must be positive power of 2\n");
		return -EINVAL;
	}

	memset(cq, 0, sizeof(*cq));
	cq->buf = attrs->buffer;
	cq->entry_size = attrs->entry_size;
	cq->num_entries = attrs->num_entries;
	cq->queue_mask = attrs->num_entries - 1;
	cq->queue_size_shift = __builtin_ctz(attrs->num_entries);
	cq->phase = 1;

	return 0;
}

static int efa_cuda_init_qp_v1(efa_cuda_qp_v1 *qp, efa_cuda_qp_attrs_v1 *attrs,
			       uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared_v1(attrs, sizeof(*attrs), inlen) ||
	    attrs->reserved) {
		printf("Incompatible attributes struct\n");
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->sq_num_entries) != 1 ||
	    __builtin_popcount(attrs->rq_num_entries) != 1) {
		printf("SQ and RQ sizes must be positive powers of 2\n");
		return -EINVAL;
	}

	memset(qp, 0, sizeof(*qp));

	efa_cuda_init_wq_v1(&qp->sq.wq, attrs->sq_buffer, attrs->sq_doorbell,
			    attrs->sq_num_entries, attrs->sq_max_batch, 0);
	qp->sq.max_inline_data = 32;
	qp->sq.max_rdma_sges = 2;

	efa_cuda_init_wq_v1(&qp->rq.wq, attrs->rq_buffer, attrs->rq_doorbell,
			    attrs->rq_num_entries, attrs->rq_num_entries, 1);

	return 0;
}

static efa_cuda_cq_v1 *efa_cuda_create_cq_v1(efa_cuda_cq_attrs_v1 *attrs, uint32_t inlen)
{
	return efa_cuda_create_device_object_v1(attrs, inlen, efa_cuda_init_cq_v1, "cq");
}

static void efa_cuda_destroy_cq_v1(efa_cuda_cq_v1 *cq)
{
	if (cq)
		cuMemFree(reinterpret_cast<CUdeviceptr>(cq));
}

static efa_cuda_qp_v1 *efa_cuda_create_qp_v1(efa_cuda_qp_attrs_v1 *attrs, uint32_t inlen)
{
	return efa_cuda_create_device_object_v1(attrs, inlen, efa_cuda_init_qp_v1, "qp");
}

static void efa_cuda_destroy_qp_v1(efa_cuda_qp_v1 *qp)
{
	efa_cuda_destroy_cq_v1(reinterpret_cast<efa_cuda_cq_v1 *>(qp));
}

static int efa_cuda_get_version_v1(int *major, int *minor, int *subminor)
{
	if (!major || !minor || !subminor)
		return -EINVAL;

	*major = EFA_CUDA_DP_VERSION_MAJOR;
	*minor = EFA_CUDA_DP_VERSION_MINOR;
	*subminor = EFA_CUDA_DP_VERSION_SUBMINOR;

	return 0;
}

extern "C" const struct efa_cuda_dp_ops_v1 efa_cuda_dp_v1 = {
	efa_cuda_create_cq_v1,
	efa_cuda_destroy_cq_v1,
	efa_cuda_create_qp_v1,
	efa_cuda_destroy_qp_v1,
	efa_cuda_init_cq_v1,
	efa_cuda_init_qp_v1,
	efa_cuda_get_version_v1,
};
