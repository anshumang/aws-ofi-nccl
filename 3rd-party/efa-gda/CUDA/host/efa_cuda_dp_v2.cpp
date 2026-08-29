// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#include "efa_cuda_dp_v1.h"
#include "efa_cuda_dp_v2.h"
#include "efa_io_defs.h"

static const char *efa_cuda_error_string_v2(CUresult result)
{
	const char *error_string = nullptr;

	if (cuGetErrorString(result, &error_string) != CUDA_SUCCESS || !error_string)
		return "unknown CUDA error";

	return error_string;
}

static bool efa_cuda_attrs_extension_is_cleared_v2(const void *attrs, size_t attrs_size,
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

static efa_cuda_cq_attrs_v1 efa_cuda_cq_attrs_v2_to_v1(const efa_cuda_cq_attrs_v2 *attrs)
{
	efa_cuda_cq_attrs_v1 attrs_v1 = {};

	attrs_v1.comp_mask = attrs->comp_mask;
	attrs_v1.flags = attrs->flags;
	attrs_v1.buffer = attrs->buffer;
	attrs_v1.num_entries = attrs->num_entries;
	attrs_v1.entry_size = attrs->entry_size;

	return attrs_v1;
}

static int efa_cuda_init_cq_v2(efa_cuda_cq_v2 *cq, efa_cuda_cq_attrs_v2 *attrs,
			       uint32_t inlen)
{
	efa_cuda_cq_attrs_v1 attrs_v1;
	efa_cuda_cq_v1 cq_v1;
	int ret;

	if (!efa_cuda_attrs_extension_is_cleared_v2(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return -EINVAL;
	}

	attrs_v1 = efa_cuda_cq_attrs_v2_to_v1(attrs);
	ret = efa_cuda_dp_v1.init_cq(&cq_v1, &attrs_v1, sizeof(attrs_v1));
	if (!ret)
		memcpy(cq, &cq_v1, sizeof(*cq));

	return ret;
}

static void efa_cuda_init_sq_wr_ctx_v2(efa_cuda_wr_ctx_v2 *ctx,
				       efa_cuda_qp_attrs_v2 *attrs)
{
	ctx->max_inline_data = attrs->sq_max_inline_data;
	ctx->max_rdma_sges = attrs->sq_max_rdma_sges;
	ctx->wqe_size = attrs->sq_entry_size;

	if (attrs->sq_entry_size == sizeof(struct efa_io_tx_wqe_128)) {
		ctx->remote_mem_offset = offsetof(struct efa_io_tx_wqe_128,
						  data.rdma_req.remote_mem);
		ctx->local_mem_offset = offsetof(struct efa_io_tx_wqe_128,
						 data.rdma_req.local_mem);
		ctx->sgl_offset = offsetof(struct efa_io_tx_wqe_128, data.sgl);
		ctx->send_inline_data_offset = offsetof(struct efa_io_tx_wqe_128,
							data.inline_data);
		ctx->write_inline_data_offset = offsetof(struct efa_io_tx_wqe_128,
							 data.rdma_req.inline_data);
	} else {
		ctx->remote_mem_offset = offsetof(struct efa_io_tx_wqe, data.rdma_req.remote_mem);
		ctx->local_mem_offset = offsetof(struct efa_io_tx_wqe, data.rdma_req.local_mem);
		ctx->sgl_offset = offsetof(struct efa_io_tx_wqe, data.sgl);
		ctx->send_inline_data_offset = offsetof(struct efa_io_tx_wqe, data.inline_data);
		ctx->write_inline_data_offset = 0;
	}
}

static void efa_cuda_init_wq_v2(efa_cuda_wq_v2 *wq, uint8_t *buffer,
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

static int efa_cuda_init_qp_v2(efa_cuda_qp_v2 *qp, efa_cuda_qp_attrs_v2 *attrs,
			       uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared_v2(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->sq_num_entries) != 1 ||
	    __builtin_popcount(attrs->rq_num_entries) != 1) {
		printf("SQ and RQ sizes must be positive powers of 2\n");
		return -EINVAL;
	}

	if (attrs->sq_wq_caps & ~EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID_V2) {
		printf("Unexpected SQ capabilities: 0x%x\n", attrs->sq_wq_caps);
		return -EOPNOTSUPP;
	}

	if (!(attrs->sq_wq_caps & EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID_V2)) {
		printf("SQ must support 64-bit request IDs\n");
		return -EOPNOTSUPP;
	}

	if (attrs->rq_wq_caps) {
		printf("Unexpected RQ capabilities: 0x%x\n", attrs->rq_wq_caps);
		return -EOPNOTSUPP;
	}

	memset(qp, 0, sizeof(*qp));

	efa_cuda_init_wq_v2(&qp->sq.wq, attrs->sq_buffer, attrs->sq_doorbell,
			    attrs->sq_num_entries, attrs->sq_max_batch, 0);

	efa_cuda_init_sq_wr_ctx_v2(&qp->sq.wr_ctx, attrs);

	efa_cuda_init_wq_v2(&qp->rq.wq, attrs->rq_buffer, attrs->rq_doorbell,
			    attrs->rq_num_entries, attrs->rq_num_entries, 1);

	return 0;
}

static efa_cuda_cq_v2 *efa_cuda_create_cq_v2(efa_cuda_cq_attrs_v2 *attrs, uint32_t inlen)
{
	efa_cuda_cq_attrs_v1 attrs_v1;

	if (!efa_cuda_attrs_extension_is_cleared_v2(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return nullptr;
	}

	attrs_v1 = efa_cuda_cq_attrs_v2_to_v1(attrs);
	return reinterpret_cast<efa_cuda_cq_v2 *>(
		efa_cuda_dp_v1.create_cq(&attrs_v1, sizeof(attrs_v1)));
}

static void efa_cuda_destroy_cq_v2(efa_cuda_cq_v2 *cq)
{
	efa_cuda_dp_v1.destroy_cq(reinterpret_cast<efa_cuda_cq_v1 *>(cq));
}

static efa_cuda_qp_v2 *efa_cuda_create_qp_v2(efa_cuda_qp_attrs_v2 *attrs, uint32_t inlen)
{
	efa_cuda_qp_v2 host_qp;
	CUdeviceptr device_memory;
	CUresult cuda_result;
	int ret;

	ret = efa_cuda_init_qp_v2(&host_qp, attrs, inlen);
	if (ret)
		return nullptr;

	cuda_result = cuMemAlloc(&device_memory, sizeof(host_qp));
	if (cuda_result != CUDA_SUCCESS) {
		printf("Failed to allocate device memory for qp: %s\n",
		       efa_cuda_error_string_v2(cuda_result));
		return nullptr;
	}

	cuda_result = cuMemcpyHtoD(device_memory, &host_qp, sizeof(host_qp));
	if (cuda_result != CUDA_SUCCESS) {
		cuMemFree(device_memory);
		printf("Failed to copy qp to device: %s\n",
		       efa_cuda_error_string_v2(cuda_result));
		return nullptr;
	}

	return reinterpret_cast<efa_cuda_qp_v2 *>(device_memory);
}

static void efa_cuda_destroy_qp_v2(efa_cuda_qp_v2 *qp)
{
	efa_cuda_dp_v1.destroy_qp(reinterpret_cast<efa_cuda_qp_v1 *>(qp));
}

extern "C" const struct efa_cuda_dp_ops_v2 efa_cuda_dp_v2 = {
	efa_cuda_create_cq_v2,
	efa_cuda_destroy_cq_v2,
	efa_cuda_create_qp_v2,
	efa_cuda_destroy_qp_v2,
	efa_cuda_init_cq_v2,
	efa_cuda_init_qp_v2,
	efa_cuda_dp_v1.get_version,
};
