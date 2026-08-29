// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_V1_H
#define EFA_CUDA_DP_V1_H

#include <stdint.h>

#include "efa_cuda_dp_types_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Backend version 1 is frozen from amzn/efa-dp-direct commit 80516b2.
 * In particular, the final QP attribute remains reserved.
 */
struct efa_cuda_cq_attrs_v1 {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *buffer;
	uint32_t num_entries;
	uint32_t entry_size;
};

struct efa_cuda_qp_attrs_v1 {
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
	uint32_t reserved;
};

typedef struct efa_cuda_cq_attrs_v1 efa_cuda_cq_attrs_v1;
typedef struct efa_cuda_qp_attrs_v1 efa_cuda_qp_attrs_v1;

struct efa_cuda_dp_ops_v1 {
	efa_cuda_cq_v1 *(*create_cq)(efa_cuda_cq_attrs_v1 *attrs, uint32_t inlen);
	void (*destroy_cq)(efa_cuda_cq_v1 *cq);
	efa_cuda_qp_v1 *(*create_qp)(efa_cuda_qp_attrs_v1 *attrs, uint32_t inlen);
	void (*destroy_qp)(efa_cuda_qp_v1 *qp);
	int (*init_cq)(efa_cuda_cq_v1 *cq, efa_cuda_cq_attrs_v1 *attrs, uint32_t inlen);
	int (*init_qp)(efa_cuda_qp_v1 *qp, efa_cuda_qp_attrs_v1 *attrs, uint32_t inlen);
	int (*get_version)(int *major, int *minor, int *subminor);
};

extern const struct efa_cuda_dp_ops_v1 efa_cuda_dp_v1;

#ifdef __cplusplus
}
#endif

#endif
