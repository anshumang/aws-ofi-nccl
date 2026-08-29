// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_V2_H
#define EFA_CUDA_DP_V2_H

#include <stdint.h>

#include "efa_cuda_dp_types_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Backend version 2 is frozen from Gerrit commit ca1f6b0.
 */
struct efa_cuda_cq_attrs_v2 {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *buffer;
	uint32_t num_entries;
	uint32_t entry_size;
};

enum efa_cuda_wq_caps_v2 {
	EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID_V2 = 1 << 0,
};

struct efa_cuda_qp_attrs_v2 {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *sq_buffer;
	uint8_t *rq_buffer;
	uint32_t *sq_doorbell;
	uint32_t *rq_doorbell;
	uint32_t sq_num_entries;
	uint32_t sq_entry_size;
	uint32_t sq_max_batch;
	uint32_t rq_num_entries;
	uint32_t rq_entry_size;
	uint32_t sq_max_inline_data;
	uint32_t sq_max_rdma_sges;
	uint32_t sq_wq_caps;
	uint32_t rq_wq_caps;
};

typedef struct efa_cuda_cq_attrs_v2 efa_cuda_cq_attrs_v2;
typedef struct efa_cuda_qp_attrs_v2 efa_cuda_qp_attrs_v2;

struct efa_cuda_dp_ops_v2 {
	efa_cuda_cq_v2 *(*create_cq)(efa_cuda_cq_attrs_v2 *attrs, uint32_t inlen);
	void (*destroy_cq)(efa_cuda_cq_v2 *cq);
	efa_cuda_qp_v2 *(*create_qp)(efa_cuda_qp_attrs_v2 *attrs, uint32_t inlen);
	void (*destroy_qp)(efa_cuda_qp_v2 *qp);
	int (*init_cq)(efa_cuda_cq_v2 *cq, efa_cuda_cq_attrs_v2 *attrs, uint32_t inlen);
	int (*init_qp)(efa_cuda_qp_v2 *qp, efa_cuda_qp_attrs_v2 *attrs, uint32_t inlen);
	int (*get_version)(int *major, int *minor, int *subminor);
};

extern const struct efa_cuda_dp_ops_v2 efa_cuda_dp_v2;

#ifdef __cplusplus
}
#endif

#endif
