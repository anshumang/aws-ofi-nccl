// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "efa_cuda_dp.h"
#include "efa_cuda_dp_types.h"
#include "efa_cuda_dp_v2.h"

/*
 * The unversioned API aliases backend v2 from Gerrit commit ca1f6b0. These
 * assertions make the bridge fail at build time if the default types move
 * without updating it.
 */
static_assert(sizeof(efa_cuda_cq) == sizeof(efa_cuda_cq_v2),
	      "default CQ size no longer matches backend v2");
static_assert(offsetof(efa_cuda_cq, db) == offsetof(efa_cuda_cq_v2, db),
	      "default CQ db offset no longer matches backend v2");
static_assert(sizeof(efa_cuda_qp) == sizeof(efa_cuda_qp_v2),
	      "default QP size no longer matches backend v2");
static_assert(sizeof(efa_cuda_wq) == sizeof(efa_cuda_wq_v2),
	      "default WQ size no longer matches backend v2");
static_assert(sizeof(efa_cuda_wr_ctx) == sizeof(efa_cuda_wr_ctx_v2),
	      "default WR context size no longer matches backend v2");
static_assert(offsetof(efa_cuda_wr_ctx, wqe_size) ==
	      offsetof(efa_cuda_wr_ctx_v2, wqe_size),
	      "default WR context offset no longer matches backend v2");
static_assert(sizeof(efa_cuda_sq) == sizeof(efa_cuda_sq_v2),
	      "default SQ size no longer matches backend v2");
static_assert(offsetof(efa_cuda_sq, wr_ctx) == offsetof(efa_cuda_sq_v2, wr_ctx),
	      "default SQ WR context offset no longer matches backend v2");
static_assert(offsetof(efa_cuda_qp, sq) == offsetof(efa_cuda_qp_v2, sq),
	      "default QP SQ offset no longer matches backend v2");
static_assert(offsetof(efa_cuda_qp, rq) == offsetof(efa_cuda_qp_v2, rq),
	      "default QP RQ offset no longer matches backend v2");
static_assert(sizeof(efa_cuda_cq_attrs) == sizeof(efa_cuda_cq_attrs_v2),
	      "default CQ attrs size no longer matches backend v2");
static_assert(offsetof(efa_cuda_cq_attrs, entry_size) ==
	      offsetof(efa_cuda_cq_attrs_v2, entry_size),
	      "default CQ attrs entry_size offset no longer matches backend v2");
static_assert(sizeof(efa_cuda_qp_attrs) == sizeof(efa_cuda_qp_attrs_v2),
	      "default QP attrs size no longer matches backend v2");
static_assert(offsetof(efa_cuda_qp_attrs, sq_max_inline_data) ==
	      offsetof(efa_cuda_qp_attrs_v2, sq_max_inline_data),
	      "default QP attrs inline-data offset no longer matches backend v2");
static_assert(offsetof(efa_cuda_qp_attrs, rq_wq_caps) ==
	      offsetof(efa_cuda_qp_attrs_v2, rq_wq_caps),
	      "default QP attrs capability offset no longer matches backend v2");

static bool efa_cuda_attrs_extension_is_cleared(const void *attrs, size_t attrs_size,
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

static efa_cuda_cq_attrs_v2 efa_cuda_cq_attrs_to_v2(const efa_cuda_cq_attrs *attrs)
{
	efa_cuda_cq_attrs_v2 attrs_v2 = {};
	attrs_v2.comp_mask = attrs->comp_mask;
	attrs_v2.flags = attrs->flags;
	attrs_v2.buffer = attrs->buffer;
	attrs_v2.num_entries = attrs->num_entries;
	attrs_v2.entry_size = attrs->entry_size;
	return attrs_v2;
}

static efa_cuda_qp_attrs_v2 efa_cuda_qp_attrs_to_v2(const efa_cuda_qp_attrs *attrs)
{
	efa_cuda_qp_attrs_v2 attrs_v2 = {};
	attrs_v2.comp_mask = attrs->comp_mask;
	attrs_v2.flags = attrs->flags;
	attrs_v2.sq_buffer = attrs->sq_buffer;
	attrs_v2.rq_buffer = attrs->rq_buffer;
	attrs_v2.sq_doorbell = attrs->sq_doorbell;
	attrs_v2.rq_doorbell = attrs->rq_doorbell;
	attrs_v2.sq_num_entries = attrs->sq_num_entries;
	attrs_v2.sq_entry_size = attrs->sq_entry_size;
	attrs_v2.sq_max_batch = attrs->sq_max_batch;
	attrs_v2.rq_num_entries = attrs->rq_num_entries;
	attrs_v2.rq_entry_size = attrs->rq_entry_size;
	attrs_v2.sq_max_inline_data = attrs->sq_max_inline_data;
	attrs_v2.sq_max_rdma_sges = attrs->sq_max_rdma_sges;
	attrs_v2.sq_wq_caps = attrs->sq_wq_caps;
	attrs_v2.rq_wq_caps = attrs->rq_wq_caps;
	return attrs_v2;
}

struct efa_cuda_cq *efa_cuda_create_cq(struct efa_cuda_cq_attrs *attrs, uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return nullptr;
	}

	efa_cuda_cq_attrs_v2 attrs_v2 = efa_cuda_cq_attrs_to_v2(attrs);
	efa_cuda_cq_v2 *cq = efa_cuda_dp_v2.create_cq(&attrs_v2, sizeof(attrs_v2));
	return reinterpret_cast<efa_cuda_cq *>(cq);
}

void efa_cuda_destroy_cq(struct efa_cuda_cq *cq)
{
	efa_cuda_dp_v2.destroy_cq(reinterpret_cast<efa_cuda_cq_v2 *>(cq));
}

struct efa_cuda_qp *efa_cuda_create_qp(struct efa_cuda_qp_attrs *attrs, uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return nullptr;
	}

	efa_cuda_qp_attrs_v2 attrs_v2 = efa_cuda_qp_attrs_to_v2(attrs);
	efa_cuda_qp_v2 *qp = efa_cuda_dp_v2.create_qp(&attrs_v2, sizeof(attrs_v2));
	return reinterpret_cast<efa_cuda_qp *>(qp);
}

void efa_cuda_destroy_qp(struct efa_cuda_qp *qp)
{
	efa_cuda_dp_v2.destroy_qp(reinterpret_cast<efa_cuda_qp_v2 *>(qp));
}

int efa_cuda_init_cq(struct efa_cuda_cq *cq, struct efa_cuda_cq_attrs *attrs, uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return -EINVAL;
	}

	efa_cuda_cq_attrs_v2 attrs_v2 = efa_cuda_cq_attrs_to_v2(attrs);
	efa_cuda_cq_v2 cq_v2;
	int ret = efa_cuda_dp_v2.init_cq(&cq_v2, &attrs_v2, sizeof(attrs_v2));
	if (!ret)
		memcpy(cq, &cq_v2, sizeof(*cq));
	return ret;
}

int efa_cuda_init_qp(struct efa_cuda_qp *qp, struct efa_cuda_qp_attrs *attrs, uint32_t inlen)
{
	if (!efa_cuda_attrs_extension_is_cleared(attrs, sizeof(*attrs), inlen)) {
		printf("Incompatible attributes struct\n");
		return -EINVAL;
	}

	efa_cuda_qp_attrs_v2 attrs_v2 = efa_cuda_qp_attrs_to_v2(attrs);
	efa_cuda_qp_v2 qp_v2;
	int ret = efa_cuda_dp_v2.init_qp(&qp_v2, &attrs_v2, sizeof(attrs_v2));
	if (!ret)
		memcpy(qp, &qp_v2, sizeof(*qp));
	return ret;
}

int efa_cuda_get_version(int *major, int *minor, int *subminor)
{
	return efa_cuda_dp_v2.get_version(major, minor, subminor);
}
