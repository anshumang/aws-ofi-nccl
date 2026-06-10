/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin for the GIN API. Shared APIs (init, devices, listen, connect,
 * regMrSym[DmaBuf], deregMrSym, closeColl, closeListen, ginProgress, finalize)
 * are reused from the proxy-side implementations in nccl_ofi_gin_api.cpp.
 * Only the GDAKI-specific APIs (createContext/destroyContext/get_properties/
 * queryLastError) live here.
 *
 * Lifecycle-managed ctx resources are implemented as standalone owner
 * classes in rdma/gin/nccl_ofi_gin_gdaki_resources.{h,cpp}; this file
 * orchestrates them.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include "cm/nccl_ofi_cm_types.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_api.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_param.h"

/*
 * GDAKI contexts tracked by collComm so we can clean them up at
 * closeColl time. NCCL's GIN call sequence is supposed to be
 *   createContext -> destroyContext -> closeColl
 * but in practice (verified on alltoall_perf shutdown) destroyContext
 * is sometimes never called, leaving libfabric EPs / hardware counters
 * / scratch MRs open on the proxy domain. The proxy plugin then closes
 * its fabric/domain with our resources still alive, producing
 * "Failed to close fid_domain/fid_fabric: Device or resource busy"
 * warnings.
 *
 * The closeColl wrapper below sweeps any contexts created against this
 * collComm before delegating to the shared closeColl, ensuring our
 * resources are released before the proxy domain goes away.
 *
 * The map + mutex are bundled into a class so callers don't have to
 * remember to take the lock when manipulating the map.
 */
class gdaki_context_registry {
public:
	/** Register a newly-created context against its collComm. */
	void add(void *collComm, nccl_ofi_gin_gdaki_context *ctx)
	{
		std::lock_guard<std::mutex> lock(mu);
		map[collComm].push_back(ctx);
	}

	/**
	 * Deregister a context from whichever collComm bucket it lives in.
	 * Called from destroyContext, which is invoked either by NCCL or
	 * by our own closeColl sweep — both paths must remove the entry
	 * to avoid a double-free.
	 */
	void remove(nccl_ofi_gin_gdaki_context *ctx)
	{
		std::lock_guard<std::mutex> lock(mu);
		for (auto it = map.begin(); it != map.end(); ) {
			auto &vec = it->second;
			vec.erase(std::remove(vec.begin(), vec.end(), ctx), vec.end());
			if (vec.empty()) {
				it = map.erase(it);
			} else {
				++it;
			}
		}
	}

	/**
	 * Atomically take and remove all contexts for a given collComm.
	 * Used by closeColl to sweep any contexts that NCCL forgot to
	 * destroyContext.
	 */
	std::vector<nccl_ofi_gin_gdaki_context *> take_all(void *collComm)
	{
		std::lock_guard<std::mutex> lock(mu);
		auto it = map.find(collComm);
		if (it == map.end()) {
			return {};
		}
		auto leftover = std::move(it->second);
		map.erase(it);
		return leftover;
	}

private:
	std::mutex mu;
	std::unordered_map<void *, std::vector<nccl_ofi_gin_gdaki_context *>> map;
};

static gdaki_context_registry gdaki_contexts;

static ncclResult_t nccl_ofi_gin_gdaki_get_properties(int dev, ncclNetProperties_v12_t *props)
{
	nccl_ofi_properties_t ofi_properties;
	ncclResult_t ret = nccl_net_ofi_get_properties(dev, &ofi_properties);
	if (ret != ncclSuccess) {
		return ret;
	}

	props->name = ofi_properties.name;
	props->pciPath = ofi_properties.pci_path;
	props->guid = ofi_properties.guid;
	props->ptrSupport = NCCL_PTR_HOST;
	if (ofi_properties.hmem_support) {
		props->ptrSupport |= NCCL_PTR_CUDA;
	}
	if (ofi_properties.dmabuf_support) {
		props->ptrSupport |= NCCL_PTR_DMABUF;
	}

	props->regIsGlobal = ofi_properties.regIsGlobal;
	props->forceFlush = 0;
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	/* This should not matter for GDAKI, but for completeness it should be set to
	* ofi_properties.max_group_receives once the perf regression in proxy mode
	* is resolved
	*/
	props->maxRecvs = 1;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;
	props->maxMultiRequestSize = 1;
	props->railId = -1;
	props->planeId = -1;

	return ncclSuccess;
}

/*
 * Read this rank's libfabric EP address into a per-rank slot of an
 * allgather buffer, then allgather across the team. Returns the buffer.
 *
 * Used by createContext for both the data endpoint and each
 * signal/counter endpoint.
 *
 * Throws std::runtime_error on fi_getname / allgather failure.
 */
static std::vector<uint8_t> allgather_ep_addr(struct fid_ep *ep,
					      nccl_ofi_rdma_gin_put_comm *put_comm,
					      int nranks, int rank,
					      size_t ep_addr_len)
{
	std::vector<uint8_t> all_addrs(nranks * ep_addr_len, 0);
	size_t addrlen = ep_addr_len;
	int ret = fi_getname(&ep->fid, &all_addrs[rank * ep_addr_len], &addrlen);
	if (ret != 0) {
		throw std::runtime_error("fi_getname failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (put_comm->get_ag_comm().all_gather(all_addrs.data(), ep_addr_len) != 0) {
		throw std::runtime_error("allgather of ep addresses failed");
	}
	return all_addrs;
}

/*
 * Allocate the per-context signal-only scratch buffer, register it on the
 * proxy domain, and allgather (addr, rkey) across ranks. The kernel uses
 * this for net.signal()-style writes that have no payload (hasWins=false,
 * bytes=0) — EFA still needs a registered remote address to bump the
 * receiver's FI_REMOTE_WRITE counter.
 *
 * Throws std::runtime_error on any libfabric / allgather failure; the
 * caller's catch handler unwinds via the ctx's RAII members.
 */
static void setup_scratch_buffer(nccl_ofi_gin_gdaki_context *ctx,
				 nccl_ofi_rdma_gin_put_comm *put_comm,
				 int nranks, int rank)
{
	constexpr size_t kScratchBytes = sizeof(uint64_t);
	ctx->scratch_buf = calloc(1, kScratchBytes);
	if (ctx->scratch_buf == nullptr) {
		throw std::runtime_error("scratch calloc failed");
	}
	ctx->scratch_local_addr = (uint64_t)ctx->scratch_buf;

	auto &domain = put_comm->get_resources().get_ep().get_domain();

	/* Register the one shared scratch buffer on each used rail's domain.
	 * Each rail produces its own lkey, and peers register on their own
	 * rails too, so the (addr, rkey) allgather is per-rail. A logical
	 * context bound to rail r reads its scratch keys from
	 * rail_shared[r]. */
	for (uint16_t r = 0; r < ctx->effective_rails; r++) {
		struct fid_domain *dom_r = domain.get_ofi_domain(r).get();
		if (dom_r == nullptr) {
			throw std::runtime_error(
				"scratch: rail " + std::to_string(r) + " domain is null");
		}
		auto &rs = *ctx->rail_shared[r];

		struct iovec iov = { .iov_base = ctx->scratch_buf,
				     .iov_len = kScratchBytes };
		struct fi_mr_attr mr_attr = {};
		mr_attr.mr_iov = &iov;
		mr_attr.iov_count = 1;
		/* Only used as source/target of RDMA WRITE on the signal
		 * endpoints; FI_SEND / FI_RECV are not required. */
		mr_attr.access = FI_REMOTE_WRITE | FI_WRITE;
		mr_attr.iface = FI_HMEM_SYSTEM;

		int mret = fi_mr_regattr(dom_r, &mr_attr, 0, &rs.scratch_mr);
		if (mret != 0) {
			throw std::runtime_error(
				"scratch fi_mr_regattr (rail " + std::to_string(r) +
				") failed: " + std::string(fi_strerror(-mret)));
		}
		rs.scratch_lkey = (uint32_t)fi_mr_key(rs.scratch_mr);

		/* Allgather (local_addr, rail-r rkey) across ranks. */
		std::vector<uint64_t> scratch_all_addrs(nranks, 0);
		std::vector<uint32_t> scratch_all_rkeys(nranks, 0);
		scratch_all_addrs[rank] = ctx->scratch_local_addr;
		scratch_all_rkeys[rank] = rs.scratch_lkey;

		if (put_comm->get_ag_comm().all_gather(
			    scratch_all_addrs.data(), sizeof(uint64_t)) != 0) {
			throw std::runtime_error("scratch allgather addrs failed");
		}
		if (put_comm->get_ag_comm().all_gather(
			    scratch_all_rkeys.data(), sizeof(uint32_t)) != 0) {
			throw std::runtime_error("scratch allgather rkeys failed");
		}

		rs.scratch_remote_addrs_buf.allocate(nranks);
		rs.scratch_remote_rkeys_buf.allocate(nranks);
		for (int i = 0; i < nranks; i++) {
			rs.scratch_remote_addrs_buf.host[i] = scratch_all_addrs[i];
			rs.scratch_remote_rkeys_buf.host[i] = scratch_all_rkeys[i];
		}
		rs.scratch_remote_addrs_buf.commit();
		rs.scratch_remote_rkeys_buf.commit();
	}
}

/*
 * Set up the PutValue source slot pool, shared across every logical
 * context's data endpoint and signal/counter (sc) endpoints.
 *
 * EFA's RDMA_WRITE WQE cannot use inline data, so PutValue stages each
 * value through a registered local source slot then RDMA-writes the
 * slot to the user's destination. The same WQE arrival on the receiver's
 * sc_endpoint bumps that endpoint's FI_REMOTE_WRITE counter, giving us
 * value-and-signal in one WQE; routing matches Put.
 *
 * No QP/EP is opened here: the WQE rides on whichever endpoint the
 * kernel selects (data or sc). The pool is one contiguous GPU-VMM
 * region registered as a single FI_HMEM_CUDA / FI_MR_DMABUF MR, sized
 * to the sum over all contexts of (data.sq_size + sum sc_endpoints[i].
 * sq_size) slots. Per-endpoint slice descriptors are uploaded so the
 * kernel can locate its slot range without sharing an allocator across
 * endpoints.
 *
 * Must be called after every context's data and sc_endpoints have been
 * populated so each sq_size is finalized.
 */
static void setup_putvalue_pool(nccl_ofi_gin_gdaki_context *ctx,
				nccl_ofi_rdma_gin_put_comm *put_comm)
{
	const int nContexts = ctx->nContexts;

	/* Total pool size = sum over all contexts of
	 *   data[c].sq_size + sum_i(sc_endpoints[c][i].sq_size).
	 * Per-endpoint slice sizes are implied by each endpoint's sq_size; we
	 * don't need a separate size array. */
	uint64_t total_slots = 0;
	for (int c = 0; c < nContexts; c++) {
		const uint32_t data_sq_size = ctx->data[c]->base.sq_size;
		if (data_sq_size == 0) {
			throw std::runtime_error(
				"putvalue: data endpoint sq_size is zero (ctx " +
				std::to_string(c) + ")");
		}
		total_slots += data_sq_size;
		const int nSc = (int)ctx->sc_endpoints[c].size();
		for (int i = 0; i < nSc; i++) {
			uint32_t sz = ctx->sc_endpoints[c][i]->base.sq_size;
			if (sz == 0) {
				throw std::runtime_error(
					"putvalue: sc_endpoint sq_size is zero (ctx " +
					std::to_string(c) + ", sc " + std::to_string(i) + ")");
			}
			total_slots += sz;
		}
	}

	ctx->putvalue_slot_size = NCCL_OFI_GDAKI_PUTVALUE_SLOT_SIZE;
	const size_t requested_bytes = (size_t)total_slots * ctx->putvalue_slot_size;

	/* Allocate GPU memory via VMM (cuMemCreate + cuMemMap with
	 * gpuDirectRDMACapable) so we can export a DMA-BUF for libfabric
	 * MR registration. The actual size returned is rounded up to the
	 * VMM granularity (typically 2 MiB on B200) — store that back so
	 * vmm_free in the destructor passes the correct size. */
	void *gpu_pool = nullptr;
	size_t actual_size = 0;
	if (nccl_net_ofi_gpu_vmm_alloc(&gpu_pool, requested_bytes, &actual_size) != 0) {
		throw std::runtime_error("putvalue gpu_vmm_alloc failed");
	}
	ctx->putvalue_buf = gpu_pool;
	ctx->putvalue_pool_bytes = actual_size;

	/* Get DMA-BUF fd. The DMA-BUF must cover the full mapped region
	 * (rounded to VMM granularity), not just the bytes we use. */
	int pv_fd = -1;
	size_t pv_fd_offset = 0;
	if (nccl_net_ofi_gpu_get_dma_buf_fd(gpu_pool, actual_size, &pv_fd, &pv_fd_offset) != 0) {
		nccl_net_ofi_gpu_vmm_free(gpu_pool, actual_size);
		ctx->putvalue_buf = nullptr;
		throw std::runtime_error("putvalue get_dma_buf_fd failed");
	}
	ctx->putvalue_dmabuf_fd = pv_fd;

	/* CUDA device id for FI_HMEM_CUDA */
	int cuda_dev = 0;
	if (nccl_net_ofi_get_gpu_device_for_addr(gpu_pool, &cuda_dev) != 0) {
		throw std::runtime_error("putvalue get_gpu_device_for_addr failed");
	}

	ctx->putvalue_local_addr = (uint64_t)gpu_pool;

	/* Register the one shared pool on each used rail's domain. The
	 * pool's GPU VA (and hence every endpoint's slice base) is the same
	 * across rails; only the lkey differs per rail. A logical context
	 * bound to rail r reads its putvalue lkey from rail_shared[r]. */
	auto &domain = put_comm->get_resources().get_ep().get_domain();
	for (uint16_t r = 0; r < ctx->effective_rails; r++) {
		struct fid_domain *dom_r = domain.get_ofi_domain(r).get();
		if (dom_r == nullptr) {
			throw std::runtime_error(
				"putvalue: rail " + std::to_string(r) + " domain is null");
		}
		auto &rs = *ctx->rail_shared[r];

		struct fi_mr_dmabuf pv_dmabuf = {};
		pv_dmabuf.fd        = pv_fd;
		pv_dmabuf.offset    = pv_fd_offset;
		pv_dmabuf.len       = actual_size;
		pv_dmabuf.base_addr = gpu_pool;

		struct fi_mr_attr pv_mr_attr = {};
		pv_mr_attr.dmabuf      = &pv_dmabuf;
		pv_mr_attr.iov_count   = 1;
		pv_mr_attr.access      = FI_REMOTE_WRITE | FI_WRITE | FI_SEND | FI_RECV;
		pv_mr_attr.iface       = FI_HMEM_CUDA;
		pv_mr_attr.device.cuda = cuda_dev;
		pv_mr_attr.requested_key = 0;

		int ret = fi_mr_regattr(dom_r, &pv_mr_attr, FI_MR_DMABUF, &rs.putvalue_mr);
		if (ret != 0) {
			throw std::runtime_error(
				"putvalue fi_mr_regattr (rail " + std::to_string(r) +
				", FI_MR_DMABUF, FI_HMEM_CUDA) failed: " +
				std::string(fi_strerror(-ret)));
		}
		rs.putvalue_lkey = (uint32_t)fi_mr_key(rs.putvalue_mr);
	}

	/* Assign per-endpoint slice bases across every context. Each
	 * endpoint stores its slice base on its own GPU-resident
	 * dev_endpoint_handle, so the kernel reads it from the same struct
	 * it already selects for QP / sq_lock / etc. The data endpoint
	 * stashes on host (uploaded later by populate_dev_handle); each
	 * sc_endpoint commits both its counter_dev_handle and
	 * signal_dev_handle inline. Slices are laid out context-major:
	 * for each context, the data slice followed by its sc slices. */
	uint64_t cursor = ctx->putvalue_local_addr;
	for (int c = 0; c < nContexts; c++) {
		ctx->data[c]->set_putvalue_slice_base(cursor);
		cursor += (uint64_t)ctx->data[c]->base.sq_size * ctx->putvalue_slot_size;
		const int nSc = (int)ctx->sc_endpoints[c].size();
		for (int i = 0; i < nSc; i++) {
			ctx->sc_endpoints[c][i]->set_putvalue_slice_base(cursor);
			cursor += (uint64_t)ctx->sc_endpoints[c][i]->base.sq_size
				* ctx->putvalue_slot_size;
		}
	}
}


/*
 * Fill one entry of the contiguous dev_handles[] GPU array for logical
 * context `ctx_id`. Caller (createContext) owns the GPU buffer; this
 * function only writes the host-side copy. The whole array is committed
 * to GPU memory once after every entry is filled.
 */
static void populate_dev_handle(nccl_ofi_gin_gdaki_dev_handle &h,
				const nccl_ofi_gin_gdaki_context *ctx,
				int ctx_id,
				int nranks, int rank)
{
	h.data.qp = ctx->data[ctx_id]->base.gpu_qp.dev();
	h.data.cq = ctx->data[ctx_id]->base.gpu_cq.dev();
	h.data.address_handles = ctx->data[ctx_id]->base.peers.ahs.dev;
	h.data.remote_qpns = ctx->data[ctx_id]->base.peers.qpns.dev;
	h.data.qkey = ctx->data[ctx_id]->base.peers.qkeys.dev;
	h.data.sq_lock = 0;
	h.data.local_cntr_value = ctx->data[ctx_id]->write_cntr.gpu_ptr();
	h.data.submitted_count = 0;
	h.data.sq_size = ctx->data[ctx_id]->base.sq_size;
	h.counter_handles = (ctx->nCounters > 0) ? ctx->d_counter_handles[ctx_id]->dev : nullptr;
	h.signal_handles  = (ctx->nSignals  > 0) ? ctx->d_signal_handles[ctx_id]->dev  : nullptr;
	h.nCounters = ctx->nCounters;
	h.nSignals  = ctx->nSignals;
	h.nranks = nranks;
	h.rank = rank;

	/* Multi-rail: bind this logical context to rail (ctx_id % num_rails)
	 * — Policy A. The kernel reads rail_id to select the matching
	 * per-rail mr_handle from the window; the scratch / putvalue keys
	 * below are pulled from this rail's shared registration so they're
	 * already rail-resolved. */
	const uint16_t rail_id = (uint16_t)(ctx_id % ctx->num_rails);
	const auto &rs = *ctx->rail_shared[rail_id];
	h.rail_id = rail_id;
	h.rail_pad = 0;

	h.scratch_lkey         = rs.scratch_lkey;
	h.scratch_pad          = 0;
	h.scratch_local_addr   = ctx->scratch_local_addr;
	h.scratch_remote_addrs = rs.scratch_remote_addrs_buf.dev;
	h.scratch_remote_rkeys = rs.scratch_remote_rkeys_buf.dev;
	/* PutValue slot pool. The pool is allocated unconditionally
	 * (sized off every context's data.sq_size, which is always
	 * non-zero), so we always populate these fields. The slot_size and
	 * slice base are rail-independent (one pool, same GPU VA on every
	 * rail); only the lkey is per-rail (from rail_shared[rail_id]).
	 * sc_endpoint slice bases live on each sc_endpoint's
	 * counter_dev_handle / signal_dev_handle, set by
	 * gdaki_sc_endpoint::set_putvalue_slice_base in setup_putvalue_pool.
	 * No commit here — the caller commits the whole dev_handles[] array
	 * once after every entry is filled. */
	h.putvalue_lkey            = rs.putvalue_lkey;
	h.putvalue_slot_size       = (uint32_t)ctx->putvalue_slot_size;
	h.data.putvalue_slice_base = ctx->data[ctx_id]->putvalue_slice_base;
}

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
	if (collComm == nullptr || config == nullptr || ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	NCCL_OFI_INFO(NCCL_NET,
		      "gin GDAKI: createContext request: nSignals=%d nCounters=%d "
		      "nContexts=%d queueDepth=%d",
		      config->nSignals, config->nCounters,
		      config->nContexts, config->queueDepth);

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	int nranks = put_comm->get_nranks();
	int rank = put_comm->get_rank();

	/* Honor config->nContexts (NCCL's GIN API contract: createContext
	 * returns one ncclNetDeviceHandle whose .handle points at an array
	 * of nContexts per-context dev handles, indexed by ctx.contextId
	 * from the kernel). Defend against pathological zero/negative
	 * values by clamping to 1. */
	int nContexts = (config->nContexts > 0) ? config->nContexts : 1;

	/*
	 * Upfront EFA hardware-counter capacity check.
	 *
	 * Each ctx allocates 1 FI_WRITE counter on the data EP plus
	 * (FI_WRITE + FI_REMOTE_WRITE) on each of its
	 * max(nSignals, nCounters) sc EPs. When the total request exceeds
	 * the per-NIC counter budget, cntr_open_ext eventually returns
	 * ENOMEM mid-loop and we tear down a partially-built ctx via the
	 * exception path. That gives the user no actionable feedback.
	 *
	 * Compute the request up front and fail with a clear message
	 * naming the knobs the user can turn down (ginContextCount,
	 * ginSignalCount, ginCounterCount). Empirically the EFA per-rank
	 * pool ceiling on p5.b200/p6.b200 is in (164, 180]; default
	 * budget 160 leaves modest headroom and is overridable via
	 * OFI_NCCL_GDAKI_HW_CNTR_BUDGET.
	 */
	{
		const int n_sc = std::max(config->nSignals, config->nCounters);
		const uint64_t hw_cntrs_per_ctx =
			static_cast<uint64_t>(1) + 2u * static_cast<uint64_t>(n_sc);
		const uint64_t hw_cntrs_total =
			static_cast<uint64_t>(nContexts) * hw_cntrs_per_ctx;

		uint64_t budget = 256; /* default budget; override via OFI_NCCL_GDAKI_HW_CNTR_BUDGET */
		if (const char *env = std::getenv("OFI_NCCL_GDAKI_HW_CNTR_BUDGET")) {
			char *endp = nullptr;
			unsigned long long v = std::strtoull(env, &endp, 0);
			if (endp != env && *endp == '\0' && v > 0) {
				budget = v;
			} else {
				NCCL_OFI_WARN("gin GDAKI: ignoring malformed "
					      "OFI_NCCL_GDAKI_HW_CNTR_BUDGET=\"%s\", "
					      "using default %llu",
					      env, (unsigned long long)budget);
			}
		}

		if (hw_cntrs_total > budget) {
			NCCL_OFI_WARN(
			    "gin GDAKI: createContext rejected — request needs "
			    "%llu hw counters/rank "
			    "(nContexts=%d × (1 + 2 × max(nSignals=%d, nCounters=%d)=%d sc EPs)) "
			    "but per-NIC budget is %llu. "
			    "Reduce one of: ginContextCount, ginSignalCount, ginCounterCount, "
			    "or override OFI_NCCL_GDAKI_HW_CNTR_BUDGET if you know your NIC "
			    "supports more (empirical EFA ceiling: ~165–180 on p5/p6.b200).",
			    (unsigned long long)hw_cntrs_total,
			    nContexts, config->nSignals, config->nCounters, n_sc,
			    (unsigned long long)budget);
			return ncclSystemError;
		}
	}

	/*
	 * The ctx holds every lifecycle-managed resource. Per-logical-context
	 * state is held in vectors of size nContexts; the device-visible
	 * dev_handles[] array is contiguous so the kernel can index by
	 * ctx.contextId.
	 *
	 * RAII unwinds members in reverse declaration order if anything
	 * throws below.
	 */
	auto ctx = std::unique_ptr<nccl_ofi_gin_gdaki_context>(
		new (std::nothrow) nccl_ofi_gin_gdaki_context());
	if (ctx == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed to allocate ctx");
		return ncclSystemError;
	}
	ctx->nContexts = nContexts;
	ctx->nranks = nranks;
	ctx->rank = rank;
	ctx->nSignals = config->nSignals;
	ctx->nCounters = config->nCounters;

	ncclNetDeviceHandle_v11_t *dev_handle_out = nullptr;

	try {
		/*
		 * Step 1: Reuse the proxy plugin's libfabric domains, one per
		 * rail (EFA NIC).
		 *
		 * On libfabric 2.4+ the proxy plugin selects the "efa-direct"
		 * fabric (see nccl_ofi_ofiutils_get_providers +
		 * prov_filter_by_match against the first entry, which is
		 * efa-direct). Each rail's domain exposes FI_EFA_GDA_OPS.
		 * Reusing them ensures MR keys registered via extGin->regMrSym
		 * are valid on the endpoints we open here.
		 *
		 * Multi-rail (Policy A): logical context c is bound to rail
		 * c % num_rails, so its data + sc endpoints open on that rail's
		 * domain. effective_rails = min(nContexts, num_rails) is how
		 * many rails get used (and get the shared scratch/putvalue MRs).
		 */
		auto &gin_ep = put_comm->get_resources().get_ep();
		auto &domain = gin_ep.get_domain();
		uint16_t num_rails = gin_ep.get_num_rails();
		if (num_rails == 0) {
			throw std::runtime_error("gin endpoint reports zero rails");
		}
		if (num_rails > NCCL_OFI_GDAKI_MAX_RAILS) {
			NCCL_OFI_INFO(NCCL_NET,
				      "gin GDAKI: capping num_rails %u -> %d (MAX_RAILS)",
				      num_rails, NCCL_OFI_GDAKI_MAX_RAILS);
			num_rails = NCCL_OFI_GDAKI_MAX_RAILS;
		}
		ctx->num_rails = num_rails;
		ctx->effective_rails =
			(uint16_t)std::min<int>(nContexts, num_rails);
		for (uint16_t r = 0; r < ctx->effective_rails; r++) {
			ctx->rail_shared[r] = std::make_unique<gdaki_rail_shared>();
		}

		auto *plugin = nccl_net_ofi_get_plugin();
		auto *device = plugin->get_device(put_comm->get_dev());
		if (device == nullptr) {
			throw std::runtime_error("get_device returned null");
		}

		/* Per-rail GDA ops table and fi_info, indexed by rail id. Open
		 * FI_EFA_GDA_OPS on each used rail's domain (used by data EPs to
		 * bind the FI_WRITE counter, data.populate(), and sc EPs). */
		struct fi_efa_ops_gda *gda_ops_rail[NCCL_OFI_GDAKI_MAX_RAILS] = {};
		struct fi_info *proxy_info_rail[NCCL_OFI_GDAKI_MAX_RAILS] = {};
		for (uint16_t r = 0; r < ctx->effective_rails; r++) {
			struct fid_domain *dom_r = domain.get_ofi_domain(r).get();
			if (dom_r == nullptr) {
				throw std::runtime_error(
					"rail " + std::to_string(r) + " domain is null");
			}
			struct fi_info *info_r = device->get_ofi_info(r);
			if (info_r == nullptr) {
				throw std::runtime_error(
					"rail " + std::to_string(r) + " fi_info is null");
			}
			struct fi_efa_ops_gda *ops_r = nullptr;
			int ret = fi_open_ops(&dom_r->fid, FI_EFA_GDA_OPS, 0,
					      reinterpret_cast<void **>(&ops_r), nullptr);
			if (ret != 0 || ops_r == nullptr) {
				throw std::runtime_error(
					"fi_open_ops FI_EFA_GDA_OPS on rail " +
					std::to_string(r) + " failed "
					"(libfabric too old, or proxy selected non-efa-direct fabric): " +
					std::string(ret ? fi_strerror(-ret) : "no ops table"));
			}
			gda_ops_rail[r] = ops_r;
			proxy_info_rail[r] = info_r;
		}

		/* Pre-size all per-ctx vectors and the contiguous dev_handles[]
		 * GPU array. data / d_counter_handles / d_signal_handles hold
		 * unique_ptrs whose targets are constructed inside the per-ctx
		 * loop below; sc_endpoints holds vectors-of-unique_ptr too.
		 * Resizing here just creates the slots (default-initialized
		 * empty unique_ptrs); endpoints / GPU buffers are allocated
		 * later only after we successfully open them. */
		ctx->data.resize(nContexts);
		for (int ctx_id = 0; ctx_id < nContexts; ctx_id++)
			ctx->data[ctx_id] = std::make_unique<gdaki_data_endpoint>();
		ctx->sc_endpoints.resize(nContexts);
		ctx->d_counter_handles.resize(nContexts);
		ctx->d_signal_handles.resize(nContexts);
		for (int ctx_id = 0; ctx_id < nContexts; ctx_id++) {
			ctx->d_counter_handles[ctx_id] =
			    std::make_unique<gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle *>>();
			ctx->d_signal_handles[ctx_id] =
			    std::make_unique<gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle *>>();
		}
		ctx->dev_handles.allocate(nContexts);

		/*
		 * Step 3: Allocate the signal-only scratch buffer once per
		 * createContext (not per logical ctx). Shared across all
		 * ctxs on this rank because the buffer's contents are never
		 * read or written — 0-byte RDMA writes only tick the per-EP
		 * FI_REMOTE_WRITE counter on the receiver, and the buffer is
		 * just a registered destination address. Registered on each
		 * used rail's domain (one MR per rail).
		 */
		setup_scratch_buffer(ctx.get(), put_comm, nranks, rank);

		/*
		 * Per-context loop: build (data EP + sc EPs + counter/signal
		 * handle arrays) for each logical context. The dev_handles[]
		 * slots are filled in a second loop below, after the PutValue
		 * pool is allocated and slice bases are assigned.
		 */
		constexpr size_t ep_addr_len = MAX_EP_ADDR;
		const int n_sc = std::max(config->nSignals, config->nCounters);
		for (int ctx_id = 0; ctx_id < nContexts; ctx_id++) {
			/* Policy A: this context's endpoints open on rail
			 * (ctx_id % num_rails)'s domain. Distinct contextIds
			 * therefore spread across the GPU's NICs. */
			const uint16_t rail_id = (uint16_t)(ctx_id % num_rails);
			struct fid_domain *ofi_domain = domain.get_ofi_domain(rail_id).get();
			struct fi_info *proxy_info = proxy_info_rail[rail_id];
			struct fi_efa_ops_gda *gda_ops = gda_ops_rail[rail_id];

			/*
			 * Step 4: Open this ctx's data endpoint on its rail's
			 * domain.
			 */
			ctx->data[ctx_id]->open(ofi_domain, proxy_info, gda_ops);

			/*
			 * Step 5: Exchange this ctx's data-EP address across the
			 * team, then complete the data EP's GPU-side build.
			 * Each ctx's data EP is its own libfabric EP, so each
			 * needs its own allgather — ctx ctx_id on rank A pairs with
			 * ctx ctx_id on rank B for cross-rank symmetric communication.
			 * Both endpoints sit on the same rail id, so they pair on
			 * the matching NIC.
			 */
			std::vector<uint8_t> all_addrs = allgather_ep_addr(
				ctx->data[ctx_id]->base.endpoint.ep, put_comm, nranks, rank, ep_addr_len);
			ctx->data[ctx_id]->populate(gda_ops, all_addrs, ep_addr_len, nranks);

			/*
			 * Step 6: Create this ctx's signal/counter endpoints
			 * (one per max(nSignals, nCounters)) on the same rail and
			 * build the GPU pointer arrays the kernel reads as
			 * dev[ctx.contextId].counter_handles[] /
			 * .signal_handles[].
			 */
			if (n_sc > 0) {
				ctx->sc_endpoints[ctx_id].reserve(n_sc);
				for (int i = 0; i < n_sc; i++) {
					ctx->sc_endpoints[ctx_id].push_back(std::make_unique<gdaki_sc_endpoint>());
					ctx->sc_endpoints[ctx_id][i]->open(ofi_domain, proxy_info, gda_ops);

					std::vector<uint8_t> sc_addrs = allgather_ep_addr(
						ctx->sc_endpoints[ctx_id][i]->base.endpoint.ep,
						put_comm, nranks, rank, ep_addr_len);

					ctx->sc_endpoints[ctx_id][i]->populate(gda_ops, sc_addrs,
									  ep_addr_len, nranks);
				}

				auto build_handle_array = [&](gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle *> &buf,
							      int count, auto get_dev_handle) {
					if (count > 0) {
						buf.allocate(count);
						for (int i = 0; i < count; i++)
							buf.host[i] = get_dev_handle(i);
						buf.commit();
					}
				};
				build_handle_array(*ctx->d_counter_handles[ctx_id], config->nCounters,
					[&](int i) { return ctx->sc_endpoints[ctx_id][i]->counter_dev_handle.dev; });
				build_handle_array(*ctx->d_signal_handles[ctx_id], config->nSignals,
					[&](int i) { return ctx->sc_endpoints[ctx_id][i]->signal_dev_handle.dev; });
			}
		}

		/*
		 * Step 7: PutValue source slot pool. Must run after every
		 * context's data and sc endpoints are populated so their
		 * sq_sizes are finalized. Allocates the shared GPU-VMM pool
		 * and assigns each endpoint's slice base (the sc endpoints
		 * commit their counter/signal dev handles inline; the data
		 * endpoint's slice base is stashed for populate_dev_handle).
		 */
		setup_putvalue_pool(ctx.get(), put_comm);

		/*
		 * Step 8: Fill every context's slot in dev_handles[]. Done in
		 * a second pass because populate_dev_handle reads the data
		 * endpoint's putvalue_slice_base, which setup_putvalue_pool
		 * only assigns once all sq_sizes are known. Don't commit per
		 * entry — the whole array is committed once below.
		 */
		for (int ctx_id = 0; ctx_id < nContexts; ctx_id++) {
			populate_dev_handle(ctx->dev_handles.host[ctx_id], ctx.get(),
					    ctx_id, nranks, rank);
		}

		/* Commit the whole dev_handles[] array (host → GPU) once. */
		ctx->dev_handles.commit();

		/*
		 * Step 8: Publish the host-side ncclNetDeviceHandle_v11_t.
		 * The kernel indexes dev_handles[ctx.contextId] to pick the
		 * per-ctx state. .size is the size of one entry, per the
		 * NCCL GIN device-handle contract.
		 */
		dev_handle_out = static_cast<ncclNetDeviceHandle_v11_t *>(
			calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
		if (dev_handle_out == nullptr) {
			throw std::runtime_error("calloc ncclNetDeviceHandle failed");
		}
		dev_handle_out->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
		dev_handle_out->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
		dev_handle_out->handle = ctx->dev_handles.dev;
		dev_handle_out->size = sizeof(nccl_ofi_gin_gdaki_dev_handle);
		dev_handle_out->needsProxyProgress = 0;

		NCCL_OFI_INFO(NCCL_NET,
			      "gin GDAKI: createContext done (nranks=%d rank=%d "
			      "nSignals=%d nCounters=%d nContexts=%d num_rails=%u "
			      "effective_rails=%u)",
			      nranks, rank,
			      config->nSignals, config->nCounters, nContexts,
			      ctx->num_rails, ctx->effective_rails);

		*ginCtx = ctx.release();
		*devHandle = dev_handle_out;

		gdaki_contexts.add(collComm,
				   static_cast<nccl_ofi_gin_gdaki_context *>(*ginCtx));

		return ncclSuccess;

	} catch (const std::exception &e) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed: %s", e.what());
		free(dev_handle_out);
		/* unique_ptr destructor runs ctx's member destructors in
		 * reverse declaration order, unwinding whatever was built. */
		return ncclSystemError;
	}
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
	/* The ctx's member destructors tear down every owned resource in
	 * reverse construction order: device handle + per-peer tables +
	 * GPU QP/CQ first, then the MMIO BAR mappings, finally the
	 * libfabric endpoint (after the BAR mappings are gone, as
	 * required). */
	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx);

	/* Deregister from the per-collComm tracking map (if present).
	 * destroyContext may be called by NCCL or by our own closeColl
	 * sweep — either path needs to remove the entry to avoid a
	 * double-free. */
	gdaki_contexts.remove(ctx);

	delete ctx;
	return ncclSuccess;
}

/*
 * GDAKI closeColl: sweep any GDAKI contexts that NCCL forgot to
 * destroyContext for this collComm, then delegate to the shared
 * closeColl. See gdaki_contexts comment above for rationale.
 */
static ncclResult_t nccl_ofi_gin_gdaki_closeColl(void *collComm)
{
	auto leftover = gdaki_contexts.take_all(collComm);

	for (auto *ctx : leftover) {
		nccl_ofi_gin_gdaki_destroyContext(ctx);
	}

	return nccl_ofi_gin_closeColl(collComm);
}

/*
 * GDAKI regMrSym:
 *
 *   1. Call the shared proxy regMrSym. Because createContext opens the
 *      GDAKI endpoint on the same fid_domain that the proxy plugin
 *      registered the MR on, the resulting keys are directly usable
 *      on the GDAKI endpoint; no second registration is needed.
 *   2. Reach through the plugin mhandle to obtain the underlying fid_mr*.
 *   3. Query the local key via gda_ops->get_mr_lkey().
 *   4. Read the plugin's already-allgathered per-peer rkeys from
 *      mhandle->remote_mr[i].mr_key[0] — no second MPI allgather needed.
 *   5. Package lkey + rkeys[nranks] into an nccl_ofi_gin_gdaki_mr_handle
 *      (the layout declared in nccl_ofi_gin_gdaki_dev.h) and return it via
 *      ginHandle. The GPU kernel reads lkey for local SGEs and rkeys[peer]
 *      for remote RDMA writes.
 *
 * The device-visible handle owns no libfabric resources — the underlying
 * fid_mr is owned by the proxy regMrSym path and torn down by its dereg.
 * We stash it on the mhandle's gin_device_handle field so deregMrSym can
 * free it with only the mhandle: NCCL's ncclGinDeregister does not pass
 * ginHandle back to deregMrSym.
 */
static ncclResult_t nccl_ofi_gin_gdaki_regMrSym(void *collComm, void *data, size_t size,
						int type, uint64_t mrFlags,
						void **mhandle, void **ginHandle)
{
	/* Step 1: delegate to the shared proxy regMrSym for the actual
	 * memory registration and per-peer rkey allgather. */
	ncclResult_t nret = nccl_ofi_gin_regMrSym(collComm, data, size, type,
						  mrFlags, mhandle, ginHandle);
	if (nret != ncclSuccess) {
		return nret;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	int nranks = put_comm->get_nranks();
	auto &gin_ep = put_comm->get_resources().get_ep();
	auto &domain = gin_ep.get_domain();
	uint16_t num_rails = gin_ep.get_num_rails();
	if (num_rails > NCCL_OFI_GDAKI_MAX_RAILS) {
		num_rails = NCCL_OFI_GDAKI_MAX_RAILS;
	}

	/* Step 2: reach through the plugin mhandle. The proxy regMrSym has
	 * registered this window's memory on every rail's domain, so
	 * sym->local_handle->get_mr(r) is the fid_mr on rail r and
	 * sym->remote_mr[i].mr_key[r] is peer i's rkey on rail r. */
	auto *sym = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(*mhandle);

	/* Step 3: allocate the window + its per-rail mr_handles as ONE
	 * contiguous block.
	 *
	 * The device dereferences this handle directly from the kernel
	 * (the returned pointer reaches the GPU as the ncclGinWindow_t
	 * argument and is read via UVA). A single self-contained allocation
	 * is therefore required — if per_rail[r] pointed at separate
	 * calloc'd blocks, the GPU could read the pointer value but not the
	 * pointed-to handle. So we lay out:
	 *
	 *   [ mr_window ][ mr_handle+peers (rail 0) ][ mr_handle+peers (rail 1) ] ...
	 *
	 * and point per_rail[r] INTO this same block. Every field the
	 * kernel touches lives in one contiguous host allocation, exactly
	 * like the single-handle layout used before multi-rail. */
	const size_t handle_size = sizeof(nccl_ofi_gin_gdaki_mr_handle) +
				   (size_t)nranks * sizeof(nccl_ofi_gin_gdaki_mr_peer);
	/* mr_handle contains a uint64_t, so 8-byte align each sub-handle.
	 * sizeof(mr_window) and handle_size are already multiples of 8. */
	const size_t window_size = sizeof(nccl_ofi_gin_gdaki_mr_window) +
				   (size_t)num_rails * handle_size;

	auto *block = static_cast<uint8_t *>(calloc(1, window_size));
	if (block == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: calloc for mr_window block failed");
		return ncclSystemError;
	}
	auto *window = reinterpret_cast<nccl_ofi_gin_gdaki_mr_window *>(block);

	for (uint16_t r = 0; r < num_rails; r++) {
		/* Open GDA ops on rail r's domain and query rail r's lkey.
		 * fi_open_ops is cheap (returns a static ops table). */
		struct fid_domain *dom_r = domain.get_ofi_domain(r).get();
		if (dom_r == nullptr) {
			NCCL_OFI_WARN("gin GDAKI: regMrSym rail %u domain is null", r);
			free(block);
			return ncclSystemError;
		}
		struct fi_efa_ops_gda *gda_ops = nullptr;
		int ret = fi_open_ops(&dom_r->fid, FI_EFA_GDA_OPS, 0,
				      reinterpret_cast<void **>(&gda_ops), nullptr);
		if (ret != 0 || gda_ops == nullptr) {
			NCCL_OFI_WARN("gin GDAKI: fi_open_ops FI_EFA_GDA_OPS on rail %u failed: %s",
				      r, ret ? fi_strerror(-ret) : "no ops table");
			free(block);
			return ncclSystemError;
		}

		struct fid_mr *mr_r = sym->local_handle->get_mr(r);
		uint32_t lkey_r = (uint32_t)gda_ops->get_mr_lkey(mr_r);

		/* per_rail[r] points INTO the same block (after the window
		 * header, at the r-th handle slot). */
		auto *gdaki_handle = reinterpret_cast<nccl_ofi_gin_gdaki_mr_handle *>(
			block + sizeof(nccl_ofi_gin_gdaki_mr_window) +
			(size_t)r * handle_size);
		gdaki_handle->lkey = lkey_r;
		gdaki_handle->nranks = nranks;
		gdaki_handle->local_addr = (uint64_t)data;
		for (int i = 0; i < nranks; i++) {
			gdaki_handle->peers[i].remote_addr =
				(uint64_t)sym->remote_mr[i].address;
			gdaki_handle->peers[i].rkey =
				(uint32_t)sym->remote_mr[i].mr_key[r];
		}
		window->per_rail[r] = gdaki_handle;
	}

	/* Stash on the mhandle so deregMrSym can free it. The mhandle and
	 * the window share a lifetime by construction. */
	sym->gin_device_handle = window;

	*ginHandle = window;
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_deregMrSym(void *collComm, void *mhandle)
{
	/* Free the contiguous window block (which embeds every per-rail
	 * mr_handle). The mhandle's gin_device_handle points at plain heap
	 * memory owned by the GDAKI regMrSym above; the underlying fid_mrs
	 * are torn down by the shared deregMrSym call that follows. */
	auto *sym = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(mhandle);
	free(sym->gin_device_handle);
	sym->gin_device_handle = nullptr;

	return nccl_ofi_gin_deregMrSym(collComm, mhandle);
}

static ncclResult_t nccl_ofi_gin_gdaki_queryLastError(void *ginCtx, bool *hasError)
{
	*hasError = false;
	return ncclSuccess;
}

/*
 * GDAKI plugin. Shared APIs are wired directly from nccl_ofi_gin_api.cpp;
 * GDAKI-specific ones above. iput/iputSignal/iget/iflush/test are nullptr —
 * no CPU involvement in GDAKI mode.
 */
ncclGin_v13_t nccl_ofi_gin_gdaki_plugin = {
	.name = "Libfabric_GDAKI",
	.init = nccl_ofi_gin_init,
	.devices = nccl_ofi_gin_devices,
	.getProperties = nccl_ofi_gin_gdaki_get_properties,
	.listen = nccl_ofi_gin_listen,
	.connect = nccl_ofi_gin_connect,
	.createContext = nccl_ofi_gin_gdaki_createContext,
	.regMrSym = nccl_ofi_gin_gdaki_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_gdaki_deregMrSym,
	.destroyContext = nccl_ofi_gin_gdaki_destroyContext,
	.closeColl = nccl_ofi_gin_gdaki_closeColl,
	.closeListen = nccl_ofi_gin_closeListen,
	.iput = nullptr,
	.iputSignal = nullptr,
	.iget = nullptr,
	.iflush = nullptr,
	.test = nullptr,
	.ginProgress = nccl_ofi_gin_ginProgress,
	.queryLastError = nccl_ofi_gin_gdaki_queryLastError,
	.finalize = nccl_ofi_gin_finalize
};

