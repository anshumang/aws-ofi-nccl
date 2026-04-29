/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Device-visible types for the GIN GDAKI data path. This header defines the
 * layout of the device handle that is returned from createContext and
 * consumed by the kernel-side GIN Put/PutValue/Signal paths.
 *
 * The structs declared here are shared between:
 *   - host-side plugin code (createContext populates instances; destroyContext
 *     tears them down), compiled as C++17.
 *   - device-side kernel code (reads the device handle to issue work requests
 *     and poll completions), compiled as CUDA.
 *
 * Keep this header free of libfabric and plugin-internal types so it can be
 * included from both contexts.
 */

#ifndef NCCL_OFI_GIN_GDAKI_DEV_H_
#define NCCL_OFI_GIN_GDAKI_DEV_H_

#include <stdint.h>

/* Forward declarations of efa-dp-direct types. efa_cuda_dp.h is only
 * included when HAVE_EFA_DP_DIRECT is set, because the header pulls in
 * <cuda_runtime.h>. Device code that uses these fields must itself include
 * efa_cuda_dp.h. */
struct efa_cuda_qp;
struct efa_cuda_cq;

/* Forward declaration of per-context counter/signal handles. Populated and
 * consumed only when nSignals > 0 or nCounters > 0.
 * Kept here so the device handle layout stays stable across follow-up
 * patches that enable signals and counters. */
struct nccl_ofi_gin_dev_counter_handle;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Device-visible handle returned from createContext.
 *
 * This struct is allocated in GPU memory. The pointer is stored in
 * ncclNetDeviceHandle_v11_t::handle and passed to device code, which
 * dereferences it directly on the GPU.
 *
 * All member pointers refer to GPU-accessible memory.
 */
struct nccl_ofi_gin_gdaki_dev_handle {
	/* efa-dp-direct QP object (in GPU memory). */
	struct efa_cuda_qp *qp;

	/* efa-dp-direct CQ object (in GPU memory). */
	struct efa_cuda_cq *cq;

	/* Per-peer address handle numbers, indexed by rank. [nranks] in GPU mem. */
	uint16_t *address_handles;

	/* Per-peer remote QP numbers, indexed by rank. [nranks] in GPU mem. */
	uint16_t *remote_qpns;

	/* Per-peer Q keys, indexed by rank. [nranks] in GPU mem. */
	uint32_t *qkey;

	/* Per-signal device handle array, [nSignals]. NULL when nSignals == 0. */
	struct nccl_ofi_gin_dev_counter_handle **signal_handles;

	/* Per-counter device handle array, [nCounters]. NULL when nCounters == 0. */
	struct nccl_ofi_gin_dev_counter_handle **counter_handles;

	/* Count of outstanding requests tracked on the device. Used by Flush.
	 * Initialized to 0. */
	uint64_t pending_reqs;

	/* Number of ranks participating in this context. Useful for kernel-
	 * side bounds checks on the per-peer arrays. */
	int32_t nranks;

	/* Rank of the local process within the context. */
	int32_t rank;
};

#ifdef __cplusplus
}
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
