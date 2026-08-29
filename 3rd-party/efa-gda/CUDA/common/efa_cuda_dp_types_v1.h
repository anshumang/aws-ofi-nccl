// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_TYPES_V1_H
#define EFA_CUDA_DP_TYPES_V1_H

#include <stdint.h>

/*
 * Backend version 1 is frozen from amzn/efa-dp-direct commit 80516b2.
 * Layout changes require a new backend version.
 */
struct efa_cuda_cq_v1 {
	uint64_t comp_mask;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t cc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_wq_v1 {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t max_batch;
	uint32_t wqes_pending;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	uint32_t pc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_rq_v1 {
	struct efa_cuda_wq_v1 wq;
};

struct efa_cuda_sq_v1 {
	struct efa_cuda_wq_v1 wq;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
};

struct efa_cuda_qp_v1 {
	uint64_t comp_mask;
	struct efa_cuda_sq_v1 sq;
	struct efa_cuda_rq_v1 rq;
};

typedef struct efa_cuda_cq_v1 efa_cuda_cq_v1;
typedef struct efa_cuda_wq_v1 efa_cuda_wq_v1;
typedef struct efa_cuda_rq_v1 efa_cuda_rq_v1;
typedef struct efa_cuda_sq_v1 efa_cuda_sq_v1;
typedef struct efa_cuda_qp_v1 efa_cuda_qp_v1;

#endif
