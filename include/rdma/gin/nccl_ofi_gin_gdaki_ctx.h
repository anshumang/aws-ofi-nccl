/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Host-side context definition for the GIN GDAKI data path.
 *
 * This header is plugin-internal. It must stay separate from the device-
 * visible header (nccl_ofi_gin_gdaki_dev.h) because it references libfabric
 * fids that are not usable from device code.
 */

#ifndef NCCL_OFI_GIN_GDAKI_CTX_H_
#define NCCL_OFI_GIN_GDAKI_CTX_H_

#include "config.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

/**
 * Host-side state associated with a single createContext call.
 *
 * createContext returns a pointer to this struct as the opaque ginCtx.
 * destroyContext consumes it to tear everything down.
 *
 * The device handle pointer (d_handle) points to a GPU memory allocation
 * that mirrors the public nccl_ofi_gin_gdaki_dev_handle layout; it is the
 * same pointer exposed to the device via ncclNetDeviceHandle_v11_t::handle.
 */
struct nccl_ofi_gin_gdaki_context {
	/* efa-direct libfabric resources created for this context. */
	struct fid_fabric *ofi_fabric;
	struct fid_domain *ofi_domain;
	struct fid_ep *ofi_ep;
	struct fid_av *ofi_av;
	struct fid_cq *ofi_cq;

	/* The fi_info used to open the above resources. Retained for teardown
	 * and for any later attribute queries. */
	struct fi_info *ofi_info;

	/* Memory registration for the SQ buffer (GPU memory). */
	struct fid_mr *sq_mr;
	void *sq_buffer_dev;
	size_t sq_buffer_size;

	/* Memory registration for the CQ buffer (GPU memory). */
	struct fid_mr *cq_mr;
	void *cq_buffer_dev;
	size_t cq_buffer_size;

	/* Pointer to the GPU-memory-resident device handle. Also exposed to
	 * device code via ncclNetDeviceHandle_v11_t. */
	struct nccl_ofi_gin_gdaki_dev_handle *d_handle;

	/* GPU-resident QP and CQ objects, stored here for teardown. */
	void *d_qp;
	void *d_cq;

	/* GPU memory allocations backing the per-peer arrays pointed to by
	 * d_handle. Held here so destroyContext can free them. */
	uint16_t *address_handles_dev;
	uint16_t *remote_qpns_dev;
	uint32_t *qkey_dev;

	/* Config fields stashed for later use when signals and counters
	 * are enabled. Unused until then. */
	int nSignals;
	int nCounters;

	/* Cached identifiers (copied from the backing comm for convenience). */
	int nranks;
	int rank;
};

#endif /* NCCL_OFI_GIN_GDAKI_CTX_H_ */
