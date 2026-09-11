/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Host-side plugin types for the GIN GDAKI data path. This header defines
 * the layout of the device handle that createContext populates in GPU
 * memory and destroyContext tears down.
 *
 * This header is plugin-internal. NCCL does not include it; it hand-mirrors
 * these structs in nccl_device/gin/efa_gda/gin_efa_gda_dev.h. This header
 * exists so that both createContext (which builds the device handle on the
 * host) and destroyContext (which frees it) agree on the GIN-specific
 * struct layout.
 *
 * The QP and CQ are opaque handles: the pointee layout is whichever frozen
 * layout the context's backendVersion names, so only code compiled for a known
 * version may cast and dereference them. The plugin itself never does -- it
 * only passes them through to this struct. NCCL's mirror declares them void*,
 * which is layout-identical.
 *
 * Keep this header free of libfabric and plugin-internal transport types.
 * It is a stable contract for the GPU-memory layout.
 */

#ifndef NCCL_OFI_GIN_GDAKI_DEV_H_
#define NCCL_OFI_GIN_GDAKI_DEV_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-slot stride (bytes) of the PutValue source pool. PutValue's T
 * is asserted by the kernel template to be <= 8 bytes; using 8 lets
 * any T fit in one slot regardless of alignment. */
#define NCCL_OFI_GDAKI_PUTVALUE_SLOT_SIZE 8

/**
 * Per-peer MR metadata used by EFA GDA WQE construction.
 *
 * EFA uses FI_MR_VIRT_ADDR, so WQEs take absolute virtual addresses for
 * both local and remote buffers. The kernel passes an offset (srcOff /
 * dstOff) and we compute the absolute address by adding the base VA.
 */
struct nccl_ofi_gin_gdaki_mr_peer {
	/* Remote rank's base virtual address for this MR. */
	uint64_t remote_addr;
	/* Remote rank's rkey for this MR. */
	uint32_t rkey;
	/* Padding to keep the struct 16-byte sized for natural alignment. */
	uint32_t pad;
};

/**
 * GDAKI memory registration handle, one per rail.
 *
 * Allocated in GPU memory (the kernel dereferences it directly via the
 * ncclGinWindow_t argument). The lkey is used by the kernel for local SGEs.
 * The peers[] array holds per-rank (remote_addr, rkey) pairs used as the
 * destination of remote RDMA writes.
 *
 * regMrSym registers the window once per rail (each rail has its own
 * libfabric domain, hence its own lkey and per-peer rkey) and returns, as
 * ginHandle, an array of per-rail handle pointers:
 * (struct nccl_ofi_gin_gdaki_mr_handle *)[num_rails]. The pointer array and
 * the handles it points at live in ONE contiguous GPU allocation (the
 * pointer entries hold device addresses into that same block), so the
 * kernel can both index the array and dereference the selected handle on
 * the GPU. The kernel selects the handle for its logical context's rail by
 * indexing that array with dev->rail_id. The kernel receives the array as a
 * ncclGinWindow_t (void*).
 *
 * Layout is shared with the NCCL mirror in
 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync.
 */
struct nccl_ofi_gin_gdaki_mr_handle {
	/* Local key for this MR on the efa-direct domain. */
	uint32_t lkey;

	/* Number of ranks (size of peers[] array). */
	int32_t nranks;

	/* Local (this rank's) base virtual address for this MR. */
	uint64_t local_addr;

	/* Per-peer remote metadata. */
	struct nccl_ofi_gin_gdaki_mr_peer peers[];
};

/* Maximum number of rails (EFA NICs) per GPU the GDAKI path supports.
 * p5en has 2 NICs/GPU; p6-b200 has 1. Used to size per-rail arrays
 * during createContext / regMrSym. */
#define NCCL_OFI_GDAKI_MAX_RAILS 2

/*
 * NCCL backend versions supported by the EFA GDAKI device layout.
 */
#define NCCL_OFI_GDAKI_BACKEND_VERSION_1 1
#define NCCL_OFI_GDAKI_BACKEND_VERSION_2 2

/*
 * Range of NCCL backend versions whose efa-dp-direct QP/CQ layouts this
 * plugin can select.
 * NCCL passes the layout version it was built to drive as
 * ncclGinConfig_v14_t::backendVersion (selected from efaGdaBackendMinVersions[]);
 * createContext_v14 refuses values outside the supported range.
 *
 * backendVersion 1 maps to efa-dp-direct API major 0 (the original layout);
 * backendVersion 2 maps to API major 1 (WQE context and 64-bit request IDs).
 *
 * INVARIANT: any change to the QP/CQ byte layout MUST add a new API major in
 * efa-dp-direct, bump this value, and add the matching mapping in
 * gdaki_efa_dp_major(). The plugin must never implement or modify a versioned
 * QP/CQ layout itself.
 */
#define NCCL_OFI_GDAKI_MIN_BACKEND_VERSION NCCL_OFI_GDAKI_BACKEND_VERSION_1
#define NCCL_OFI_GDAKI_MAX_BACKEND_VERSION NCCL_OFI_GDAKI_BACKEND_VERSION_2

/* No default layout version: every GDAKI context gets one from
 * ncclGinConfig_v14_t::backendVersion, so an unset value is a bug rather than
 * something to fall back from. */
#define NCCL_OFI_GDAKI_BACKEND_VERSION_UNSET (-1)

/* Deliberately incomplete: the selected backendVersion defines the pointee
 * layout. */
struct nccl_ofi_gin_gdaki_dev_qp;
struct nccl_ofi_gin_gdaki_dev_cq;

/* req_id is echoed by the CQE. The device splits req_id into a peer
 * id and a per-peer sequence number (pseq). These three macros size the pseq
 * field and the per-peer completion bitmask that pseq indexes.
 *
 * NCCL_OFI_GDAKI_PSEQ_BITS is the number of req_id bits that hold pseq.
 * NCCL_OFI_GDAKI_PEER_WINDOW is an independent knob: it is both the maximum number
 * of writes outstanding to one peer (the cap put enforces) and the width in bits of
 * that peer's completion bitmask, so pseq wraps at PEER_WINDOW without aliasing a
 * live write. It must not exceed 2^PSEQ_BITS. NCCL_OFI_GDAKI_PEER_BITS_WORDS is
 * PEER_WINDOW expressed in 64-bit words, which is that bitmask's storage size.
 *
 * The window is deep because it caps how many writes put may have outstanding to one
 * peer: SRD completion latency has a long tail, so a window sized for typical latency
 * stalls put on the slowest completions rather than on the link. The cost is the
 * bitmask, at PEER_WINDOW/8 bytes of host memory per (context, peer).
 *
 * The host publishes PEER_WINDOW into dev_handle.peer_window at context setup.
 * The device reads it as the per-peer backpressure cap in put, and put stamps
 * each write's req_id with the same PSEQ_BITS split. */
#define NCCL_OFI_GDAKI_PSEQ_BITS 32u
#define NCCL_OFI_GDAKI_PSEQ_MASK ((1ull << NCCL_OFI_GDAKI_PSEQ_BITS) - 1ull)
#define NCCL_OFI_GDAKI_PEER_WINDOW (1u << 20)
#define NCCL_OFI_GDAKI_PEER_BITS_WORDS (NCCL_OFI_GDAKI_PEER_WINDOW / 64u)


/*
 * The device ABI is versioned as a whole, including the endpoint and
 * counter-handle layouts embedded in the top-level handle.
 *
 * v1 is the NCCL 2.31 layout. Each endpoint exposes a device-visible CQ and
 * lock, and PutValue stages through a registered source-slot pool.
 *
 * v2 is the NCCL 2.32 layout. CQs are drained by the plugin, the lock is part
 * of the efa-dp-direct QP descriptor, and the top-level handle carries the
 * host-published completion state used by FlushAsync/Wait. PutValue is inline,
 * so v2 carries no staging-pool metadata.
 */

struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v1 {
	struct nccl_ofi_gin_gdaki_dev_qp *qp;
	struct nccl_ofi_gin_gdaki_dev_cq *cq;
	uint16_t *target_address_handles;
	uint16_t *target_remote_qpns;
	uint32_t *target_qkey;
	uint32_t sq_lock;
	volatile uint64_t *local_cntr_value;
	uint64_t submitted_count;
	uint32_t sq_size;
	uint32_t putvalue_pad;
	uint64_t putvalue_slice_base;
};

struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v2 {
	struct nccl_ofi_gin_gdaki_dev_qp *qp;
	uint16_t *target_address_handles;
	uint16_t *target_remote_qpns;
	uint32_t *target_qkey;
	volatile uint64_t *local_cntr_value;
	uint64_t submitted_count;
	uint32_t sq_size;

	/*
	 * backendVersion 2 is already published with these unused legacy slots in
	 * NCCL's mirror. Keep them reserved so the polling fields below retain
	 * their frozen offsets, but do not expose or populate staging metadata.
	 */
	uint32_t reserved0;
	uint64_t reserved1;
};

struct nccl_ofi_gin_gdaki_dev_counter_handle_v1 {
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v1 base;
	volatile uint64_t *cntr_value;
	uint64_t cntr_offset;
};

struct nccl_ofi_gin_gdaki_dev_counter_handle_v2 {
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v2 base;
	volatile uint64_t *cntr_value;
	uint64_t cntr_offset;
};

struct nccl_ofi_gin_gdaki_dev_handle_v1 {
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v1 data;
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v1 pvdata;
	struct nccl_ofi_gin_gdaki_dev_counter_handle_v1 **counter_handles;
	struct nccl_ofi_gin_gdaki_dev_counter_handle_v1 **signal_handles;
	int32_t nCounters;
	int32_t nSignals;
	int32_t nranks;
	int32_t rank;
	uint32_t rail_id;
	uint32_t scratch_lkey;
	uint32_t scratch_pad;
	uint64_t scratch_local_addr;
	uint64_t *scratch_remote_addrs;
	uint32_t *scratch_remote_rkeys;
	uint32_t putvalue_lkey;
	uint32_t putvalue_slot_size;
};

struct nccl_ofi_gin_gdaki_dev_handle_v2 {
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v2 data;
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle_v2 pvdata;
	struct nccl_ofi_gin_gdaki_dev_counter_handle_v2 **counter_handles;
	struct nccl_ofi_gin_gdaki_dev_counter_handle_v2 **signal_handles;
	int32_t nCounters;
	int32_t nSignals;
	int32_t nranks;
	int32_t rank;
	uint32_t rail_id;
	uint32_t scratch_lkey;
	uint32_t scratch_pad;
	uint64_t scratch_local_addr;
	uint64_t *scratch_remote_addrs;
	uint32_t *scratch_remote_rkeys;

	/* Reserved offsets from the published v2 ABI; never staging metadata. */
	uint32_t reserved0;
	uint32_t reserved1;

	uint32_t *submitted_count_per_peer;
	volatile uint32_t *ordered_completed_count_per_peer;
	uint32_t peer_window;
	uint64_t *submitted_count_per_ctx;
	volatile uint64_t *completed_count_per_ctx;
	uint32_t cq_depth;
};

#ifdef __cplusplus
}

#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(nccl_ofi_gin_gdaki_dev_endpoint_handle_v1) == 80);
static_assert(sizeof(nccl_ofi_gin_gdaki_dev_counter_handle_v1) == 96);
static_assert(sizeof(nccl_ofi_gin_gdaki_dev_handle_v1) == 240);
static_assert(offsetof(nccl_ofi_gin_gdaki_dev_endpoint_handle_v1, cq) == 8);
static_assert(offsetof(nccl_ofi_gin_gdaki_dev_endpoint_handle_v1, putvalue_slice_base) == 72);
static_assert(offsetof(nccl_ofi_gin_gdaki_dev_handle_v1, putvalue_lkey) == 232);
static_assert(sizeof(nccl_ofi_gin_gdaki_dev_endpoint_handle_v2) == 64);
static_assert(sizeof(nccl_ofi_gin_gdaki_dev_counter_handle_v2) == 80);
static_assert(sizeof(nccl_ofi_gin_gdaki_dev_handle_v2) == 256);
static_assert(offsetof(nccl_ofi_gin_gdaki_dev_endpoint_handle_v2, local_cntr_value) == 32);
static_assert(offsetof(nccl_ofi_gin_gdaki_dev_handle_v2, submitted_count_per_peer) == 208);
static_assert(offsetof(nccl_ofi_gin_gdaki_dev_handle_v2, submitted_count_per_ctx) == 232);
#endif
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
