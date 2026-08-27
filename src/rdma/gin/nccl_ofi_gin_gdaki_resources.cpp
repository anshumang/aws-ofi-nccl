/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Resource owners for the GIN GDAKI data path. See
 * nccl_ofi_gin_gdaki_resources.h for the public declarations.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"

#include "efa_cuda_dp_v1.h"
#include "efa_cuda_dp_v2.h"

#include <errno.h>
#include <limits>

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

/*
 * Inline-data capacity requested for a backendVersion-2 endpoint.
 *
 * A 64B EFA WQE holds at most 32B of inline data and has no inline region in
 * its RDMA-write descriptor at all. Requesting more than 32B is what makes
 * libfabric create the QP with EFADV_QP_FLAGS_INLINE_WRITE, which rdma-core
 * turns into a 128B WQE.
 */
static constexpr size_t gdaki_wide_wqe_inject_size = 33;
static constexpr uint32_t gdaki_wide_wqe_inline_size = 80;
static constexpr uint32_t gdaki_max_rdma_sges = 2;

/*
 * Build fi_getinfo hints for the GDAKI endpoint.
 *
 * GDAKI states its own requirements explicitly (rather than copying
 * ref_info's attributes); ref_info is used only for the fabric /
 * domain / provider names that narrow fi_getinfo to the single
 * efa-direct provider entry the proxy already opened. This mirrors
 * the proxy plugin's get_gin_hints pattern in nccl_ofi_gin_resources.cpp.
 *
 * GDAKI does not register memory on this EP — the proxy's regMrSym
 * registers on the shared domain — and does not do fi_cq_readfrom,
 * so FI_SOURCE is not requested. FI_HMEM is still needed because the endpoint
 * is used to access GPU memory. efa-direct requires FI_CONTEXT2 per fi_efa(7).
 */
static void get_gdaki_hints(struct fi_info &hints, struct fi_info *ref_info,
			    int backend_version)
{
	hints.caps = FI_MSG | FI_RMA | FI_HMEM;
	hints.mode = FI_CONTEXT2;

	hints.ep_attr->type = FI_EP_RDM;
	hints.addr_format = FI_ADDR_EFA;

	hints.domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM |
				     FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
				     FI_MR_PROV_KEY;
	hints.domain_attr->threading = FI_THREAD_SAFE;
	hints.domain_attr->control_progress = FI_PROGRESS_AUTO;
	hints.domain_attr->data_progress = FI_PROGRESS_AUTO;

	/* Ask for a 128B WQE on backendVersion 2 only. */
	if (backend_version == 2) {
		hints.tx_attr->inject_size = gdaki_wide_wqe_inject_size;
	}

	/* Narrow fi_getinfo to the provider / fabric / domain the proxy
	 * already opened. Names are required to obtain exactly one result. */
	hints.fabric_attr->prov_name = strdup(ref_info->fabric_attr->prov_name);
	hints.fabric_attr->name = strdup(ref_info->fabric_attr->name);
	hints.domain_attr->name = strdup(ref_info->domain_attr->name);
}

/*
 * Obtain a GDAKI-owned fi_info via fi_getinfo, narrowed to exactly the
 * fabric / domain the proxy reference points at.
 */
static struct fi_info *get_gdaki_info(struct fi_info *ref_info, int backend_version)
{
	struct fi_info *hints = fi_allocinfo();
	if (hints == nullptr) {
		throw std::runtime_error("fi_allocinfo for GDAKI hints failed");
	}
	get_gdaki_hints(*hints, ref_info, backend_version);

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL,
			     hints, &results);
	fi_freeinfo(hints);
	if (ret != 0) {
		throw std::runtime_error("fi_getinfo for GDAKI info failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (results == nullptr) {
		throw std::runtime_error("fi_getinfo returned no GDAKI providers");
	}
	if (results->next != nullptr) {
		fi_freeinfo(results);
		throw std::runtime_error(
			"fi_getinfo returned more than one GDAKI provider; "
			"hints were not narrow enough");
	}

	return results;
}

void gdaki_fi_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     size_t cq_size, int backend_version)
{
	if (ep || cq || av || info) {
		throw std::runtime_error("gdaki_fi_endpoint: double open");
	}

	info = get_gdaki_info(ref_info, backend_version);

	struct fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = cq_size;
	int ret = fi_cq_open(domain, &cq_attr, &cq, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_cq_open on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	struct fi_av_attr av_attr = {};
	av_attr.type = FI_AV_TABLE;
	ret = fi_av_open(domain, &av_attr, &av, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_av_open on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_endpoint(domain, info, &ep, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_endpoint on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind CQ failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind AV failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_fi_endpoint::enable()
{
	int ret = fi_enable(ep);
	if (ret != 0) {
		throw std::runtime_error("fi_enable failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_fi_endpoint::bind(struct fid *fid, uint64_t flags)
{
	int ret = fi_ep_bind(ep, fid, flags);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

gdaki_gpu_qp::~gdaki_gpu_qp()
{
	if (qp == nullptr) {
		return;
	}

	/* Must pair with the table that created it; build() leaves qp null for any
	 * version without a case here, so default is unreachable. */
	switch (backend_version) {
	case 1:
		efa_cuda_dp_v1.destroy_qp(reinterpret_cast<efa_cuda_qp_v1 *>(qp));
		break;
	case 2:
		efa_cuda_dp_v2.destroy_qp(reinterpret_cast<efa_cuda_qp_v2 *>(qp));
		break;
	default:
		NCCL_OFI_WARN("gin GDAKI: cannot destroy QP descriptor for "
			      "backendVersion %d; leaking it",
			      backend_version);
		break;
	}
}

void gdaki_gpu_qp::build(int backend_version_in,
			 const struct fi_efa_wq_attr &sq_attr,
			 const struct fi_efa_wq_attr &rq_attr,
			 void *sq_buf_dev, void *sq_db_dev)
{
	/* Each create call allocates GPU memory; rebuilding would overwrite the
	 * only owned pointer and leak the previous descriptor. */
	if (qp != nullptr) {
		throw std::runtime_error("gdaki_gpu_qp: double build");
	}

	nccl_ofi_gin_gdaki_dev_qp *built = nullptr;
	switch (backend_version_in) {
	case 1: {
		efa_cuda_qp_attrs_v1 attrs = {};
		attrs.sq_buffer = static_cast<uint8_t *>(sq_buf_dev);
		attrs.rq_buffer = static_cast<uint8_t *>(rq_attr.buffer);
		attrs.sq_doorbell = static_cast<uint32_t *>(sq_db_dev);
		attrs.rq_doorbell = static_cast<uint32_t *>(rq_attr.doorbell);
		attrs.sq_num_entries = sq_attr.num_entries;
		attrs.sq_entry_size = sq_attr.entry_size;
		attrs.sq_max_batch = sq_attr.max_batch;
		attrs.rq_num_entries = rq_attr.num_entries;
		attrs.rq_entry_size = rq_attr.entry_size;
		attrs.reserved = 0;

		efa_cuda_qp_v1 *d = efa_cuda_dp_v1.create_qp(&attrs, sizeof(attrs));
		built = reinterpret_cast<nccl_ofi_gin_gdaki_dev_qp *>(d);
		break;
	}
	case 2: {
		efa_cuda_qp_attrs_v2 attrs = {};
		attrs.sq_buffer = static_cast<uint8_t *>(sq_buf_dev);
		attrs.rq_buffer = static_cast<uint8_t *>(rq_attr.buffer);
		attrs.sq_doorbell = static_cast<uint32_t *>(sq_db_dev);
		attrs.rq_doorbell = static_cast<uint32_t *>(rq_attr.doorbell);
		attrs.sq_num_entries = sq_attr.num_entries;
		attrs.sq_entry_size = sq_attr.entry_size;
		attrs.sq_max_batch = sq_attr.max_batch;
		attrs.rq_num_entries = rq_attr.num_entries;
		attrs.rq_entry_size = rq_attr.entry_size;
		attrs.sq_max_inline_data = gdaki_wide_wqe_inline_size;
		attrs.sq_max_rdma_sges = gdaki_max_rdma_sges;
		attrs.sq_wq_caps = EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID_V2;
		attrs.rq_wq_caps = 0;

		efa_cuda_qp_v2 *d = efa_cuda_dp_v2.create_qp(&attrs, sizeof(attrs));
		built = reinterpret_cast<nccl_ofi_gin_gdaki_dev_qp *>(d);
		break;
	}
	default:
		throw std::runtime_error("gdaki_gpu_qp: no QP layout for backendVersion " +
					 std::to_string(backend_version_in));
	}

	if (built == nullptr) {
		throw std::runtime_error("gdaki_gpu_qp: QP descriptor build failed for "
					 "backendVersion " +
					 std::to_string(backend_version_in));
	}

	/* Set together: the destructor dispatches on backend_version. */
	qp = built;
	backend_version = backend_version_in;
}

gdaki_gpu_cq::~gdaki_gpu_cq()
{
	if (cq == nullptr) {
		return;
	}

	switch (backend_version) {
	case 1:
		efa_cuda_dp_v1.destroy_cq(reinterpret_cast<efa_cuda_cq_v1 *>(cq));
		break;
	case 2:
		efa_cuda_dp_v2.destroy_cq(reinterpret_cast<efa_cuda_cq_v2 *>(cq));
		break;
	default:
		NCCL_OFI_WARN("gin GDAKI: cannot destroy CQ descriptor for "
			      "backendVersion %d; leaking it",
			      backend_version);
		break;
	}
}

void gdaki_gpu_cq::build(int backend_version_in, const struct fi_efa_cq_attr &cq_attr)
{
	/* Each create call allocates GPU memory; rebuilding would overwrite the
	 * only owned pointer and leak the previous descriptor. */
	if (cq != nullptr) {
		throw std::runtime_error("gdaki_gpu_cq: double build");
	}

	nccl_ofi_gin_gdaki_dev_cq *built = nullptr;
	switch (backend_version_in) {
	case 1: {
		efa_cuda_cq_attrs_v1 attrs = {};
		attrs.buffer = static_cast<uint8_t *>(cq_attr.buffer);
		attrs.num_entries = cq_attr.num_entries;
		attrs.entry_size = cq_attr.entry_size;

		efa_cuda_cq_v1 *d = efa_cuda_dp_v1.create_cq(&attrs, sizeof(attrs));
		built = reinterpret_cast<nccl_ofi_gin_gdaki_dev_cq *>(d);
		break;
	}
	case 2: {
		efa_cuda_cq_attrs_v2 attrs = {};
		attrs.buffer = static_cast<uint8_t *>(cq_attr.buffer);
		attrs.num_entries = cq_attr.num_entries;
		attrs.entry_size = cq_attr.entry_size;

		efa_cuda_cq_v2 *d = efa_cuda_dp_v2.create_cq(&attrs, sizeof(attrs));
		built = reinterpret_cast<nccl_ofi_gin_gdaki_dev_cq *>(d);
		break;
	}
	default:
		throw std::runtime_error("gdaki_gpu_cq: no CQ layout for backendVersion " +
					 std::to_string(backend_version_in));
	}

	if (built == nullptr) {
		throw std::runtime_error("gdaki_gpu_cq: CQ descriptor build failed for "
					 "backendVersion " +
					 std::to_string(backend_version_in));
	}

	cq = built;
	backend_version = backend_version_in;
}

/*
 * Sentinel for "this peer has no endpoint at this slot". A rank that did
 * not create an endpoint for a given (collective) allgather round leaves
 * its slot zero-filled; ranks are NOT required to request the same number
 * of signal/counter endpoints, so some (slot, peer) pairs have no remote
 * endpoint. A real EFA address (FI_ADDR_EFA: AH + QPN + QKEY-derived bytes)
 * is never all-zero, so an all-zero slot unambiguously means "absent" and
 * is skipped during addressing — the corresponding table entry stays 0 and
 * is never used by a correct caller (the kernel bounds each post by the
 * target's advertised count).
 */
static bool gdaki_ep_addr_is_absent(const uint8_t *addr, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		if (addr[i] != 0) {
			return false;
		}
	}
	return true;
}

void gdaki_target_addressing::populate(gdaki_fi_endpoint &endpoint,
				       const std::vector<uint8_t> &all_addrs,
				       size_t ep_addr_len, int total_slots, int nranks,
				       struct fi_efa_ops_gda *gda_ops)
{
	if (total_slots <= 0 || nranks <= 0) {
		return;
	}

	const size_t n = (size_t)total_slots * (size_t)nranks;
	ahs.allocate(n);
	qpns.allocate(n);
	qkeys.allocate(n);

	/* Build the targetSlot-major table: idx = slot * nranks + peer.
	 *
	 * `all_addrs` is the batched-allgather buffer, peer-major:
	 *     addr(peer, slot) = &all_addrs[(peer * total_slots + slot) * ep_addr_len]
	 * with slot 0 = peer's data EP and slots 1..(total_slots-1) = peer's
	 * sc EPs. We resolve every (slot, peer) through THIS endpoint's own AV
	 * (an address handle is AV-local) and store it at [slot * nranks + peer].
	 *
	 * A (slot, peer) whose peer has no endpoint at that slot (asymmetric
	 * counts leave a zero address) is skipped; the entry stays 0 and is
	 * never addressed — the kernel bounds each signalling post by the
	 * target peer's advertised signal count. */
	for (int slot = 0; slot < total_slots; slot++) {
		for (int peer = 0; peer < nranks; peer++) {
			const size_t dst_idx = (size_t)slot * (size_t)nranks + (size_t)peer;
			const size_t src_idx = (size_t)peer * (size_t)total_slots + (size_t)slot;
			const uint8_t *addr = all_addrs.data() + src_idx * ep_addr_len;

			if (gdaki_ep_addr_is_absent(addr, ep_addr_len)) {
				continue;
			}

			fi_addr_t fi_addr;
			int ret = fi_av_insert(endpoint.av, addr, 1, &fi_addr, 0, nullptr);
			if (ret != 1) {
				throw std::runtime_error(
					"target fi_av_insert failed (slot " +
					std::to_string(slot) + ", rank " +
					std::to_string(peer) + ")");
			}

			uint16_t ahn = 0, remote_qpn = 0;
			uint32_t remote_qkey = 0;
			ret = gda_ops->query_addr(endpoint.ep, fi_addr, &ahn,
						  &remote_qpn, &remote_qkey);
			if (ret != 0) {
				throw std::runtime_error(
					"target query_addr failed (slot " +
					std::to_string(slot) + ", rank " +
					std::to_string(peer) + ")");
			}

			ahs.host[dst_idx] = ahn;
			qpns.host[dst_idx] = remote_qpn;
			qkeys.host[dst_idx] = remote_qkey;
		}
	}

	ahs.commit();
	qpns.commit();
	qkeys.commit();
}

void gdaki_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			  size_t cq_size, int backend_version)
{
	endpoint.open(domain, ref_info, cq_size, backend_version);
	endpoint.enable();
}

void gdaki_endpoint::populate(int backend_version, struct fi_efa_ops_gda *gda_ops,
			      const std::vector<uint8_t> &all_addrs,
			      size_t ep_addr_len, int total_slots, int nranks)
{
	/* Query QP and map SQ MMIO for GPU access. */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	int ret = gda_ops->query_qp_wqs(endpoint.ep, &sq_attr, &rq_attr);
	if (ret != 0)
		throw std::runtime_error("gdaki_endpoint query_qp_wqs failed: " +
					 std::string(fi_strerror(-ret)));

	if (backend_version == 2 && sq_attr.entry_size != 128) {
		throw std::runtime_error(
			"gdaki_endpoint: backendVersion 2 requires a 128-byte SQ WQE "
			"but this endpoint has " + std::to_string(sq_attr.entry_size) +
			" bytes; the device or libfabric does not support inline "
			"RDMA write (wide WQE)");
	}

	sq_buffer.map(sq_attr.buffer,
		      (size_t)sq_attr.num_entries * sq_attr.entry_size);

	/* rdma-core mmaps the doorbell MMIO region with sysconf(_SC_PAGESIZE)
	 * (see providers/efa/verbs.c). Use the plugin's cached system_page_size
	 * so our GPU-side mapping covers the same region rdma-core opened. */
	sq_doorbell.map(sq_attr.doorbell, system_page_size);

	gpu_qp.build(backend_version, sq_attr, rq_attr, sq_buffer.dev, sq_doorbell.dev);

	/* Stash SQ ring depth for the device-side SQ-overflow backpressure
	 * check. The data and pvdata gdaki_data_endpoint instances, plus each
	 * gdaki_sc_endpoint, read this via base.sq_size. entry_size is kept for
	 * createContext's log line. */
	sq_size = sq_attr.num_entries;
	sq_entry_size = sq_attr.entry_size;

	/* Query CQ and build GPU descriptor. */
	struct fi_efa_cq_attr efa_cq_attr = {};
	ret = gda_ops->query_cq(endpoint.cq, &efa_cq_attr);
	if (ret != 0)
		throw std::runtime_error("gdaki_endpoint query_cq failed: " +
					 std::string(fi_strerror(-ret)));

	gpu_cq.build(backend_version, efa_cq_attr);

	/* Build the [total_slots*nranks] target table in GPU memory. */
	targets.populate(endpoint, all_addrs, ep_addr_len, total_slots, nranks, gda_ops);
}

void gdaki_data_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			       struct fi_efa_ops_gda *gda_ops, int backend_version,
			       uint64_t cntr_flags)
{
	/* Create the counter first; it will be bound to the inner endpoint
	 * between open() and enable(). */
	local_cntr.create(gda_ops, domain);

	/* Open the inner endpoint without enable. */
	base.endpoint.open(domain, ref_info, ofi_nccl_cq_size(), backend_version);

	base.endpoint.bind(&local_cntr.get()->fid, cntr_flags);

	base.endpoint.enable();
}

void gdaki_data_endpoint::populate(int backend_version, struct fi_efa_ops_gda *gda_ops,
				   const std::vector<uint8_t> &all_addrs,
				   size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP/CQ query, MMIO map, GPU descriptors,
	 * target table, sq_size stash) to the inner endpoint. */
	base.populate(backend_version, gda_ops, all_addrs, ep_addr_len, total_slots, nranks);
}

gdaki_putvalue_v1_staging::~gdaki_putvalue_v1_staging()
{
	release();
}

void gdaki_putvalue_v1_staging::release() noexcept
{
	for (auto &registration : rail_registrations) {
		if (registration.mr != nullptr) {
			fi_close(&registration.mr->fid);
			registration.mr = nullptr;
		}
		registration.lkey = 0;
	}

	if (dmabuf_fd >= 0) {
		close(dmabuf_fd);
		dmabuf_fd = -1;
	}

	if (pool != nullptr) {
		nccl_net_ofi_gpu_vmm_free(pool, pool_bytes);
		pool = nullptr;
	}
	pool_bytes = 0;
	context_slice_bases.clear();
}

void gdaki_putvalue_v1_staging::setup(
	const std::vector<uint32_t> &context_sq_sizes,
	const std::vector<struct fid_domain *> &rail_domains)
{
	if (initialized() || dmabuf_fd >= 0 || !context_slice_bases.empty()) {
		throw std::runtime_error(
			"gdaki_putvalue_v1_staging: setup called more than once");
	}
	if (context_sq_sizes.empty()) {
		throw std::runtime_error(
			"gdaki_putvalue_v1_staging: no logical contexts");
	}
	if (rail_domains.empty() ||
	    rail_domains.size() > rail_registrations.size()) {
		throw std::runtime_error(
			"gdaki_putvalue_v1_staging: invalid active rail count");
	}
	for (size_t rail_id = 0; rail_id < rail_domains.size(); rail_id++) {
		if (rail_domains[rail_id] == nullptr) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: rail " +
				std::to_string(rail_id) + " domain is null");
		}
	}

	uint64_t total_slots = 0;
	for (size_t context_id = 0; context_id < context_sq_sizes.size();
	     context_id++) {
		const uint32_t sq_size = context_sq_sizes[context_id];
		if (sq_size == 0) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: context " +
				std::to_string(context_id) + " SQ size is zero");
		}
		if (sq_size > std::numeric_limits<uint64_t>::max() - total_slots) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: slot count overflow");
		}
		total_slots += sq_size;
	}
	if (total_slots >
	    std::numeric_limits<size_t>::max() / slot_size_bytes) {
		throw std::runtime_error(
			"gdaki_putvalue_v1_staging: pool size overflow");
	}
	const size_t requested_bytes =
		static_cast<size_t>(total_slots) * slot_size_bytes;

	try {
		context_slice_bases.resize(context_sq_sizes.size());

		void *gpu_pool = nullptr;
		size_t actual_size = 0;
		int ret = nccl_net_ofi_gpu_vmm_alloc(
			&gpu_pool, requested_bytes, &actual_size);
		pool = gpu_pool;
		pool_bytes = actual_size;
		if (ret != 0) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: gpu_vmm_alloc failed");
		}
		if (pool == nullptr || pool_bytes < requested_bytes) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: invalid VMM allocation");
		}

		size_t fd_offset = 0;
		if (nccl_net_ofi_gpu_get_dma_buf_fd(
			    pool, pool_bytes, &dmabuf_fd, &fd_offset) != 0) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: get_dma_buf_fd failed");
		}

		int cuda_device = 0;
		if (nccl_net_ofi_get_gpu_device_for_addr(
			    pool, &cuda_device) != 0) {
			throw std::runtime_error(
				"gdaki_putvalue_v1_staging: "
				"get_gpu_device_for_addr failed");
		}

		struct fi_mr_dmabuf dmabuf = {};
		dmabuf.fd = dmabuf_fd;
		dmabuf.offset = fd_offset;
		dmabuf.len = pool_bytes;
		dmabuf.base_addr = pool;

		struct fi_mr_attr mr_attr = {};
		mr_attr.dmabuf = &dmabuf;
		mr_attr.iov_count = 1;
		mr_attr.access = FI_WRITE;
		mr_attr.iface = FI_HMEM_CUDA;
		mr_attr.device.cuda = cuda_device;
		mr_attr.requested_key = 0;

		for (size_t rail_id = 0; rail_id < rail_domains.size();
		     rail_id++) {
			auto &registration = rail_registrations[rail_id];
			ret = fi_mr_regattr(
				rail_domains[rail_id], &mr_attr, FI_MR_DMABUF,
				&registration.mr);
			if (ret != 0) {
				throw std::runtime_error(
					"gdaki_putvalue_v1_staging: "
					"fi_mr_regattr on rail " +
					std::to_string(rail_id) + " failed: " +
					std::string(fi_strerror(-ret)));
			}
			registration.lkey =
				static_cast<uint32_t>(fi_mr_key(registration.mr));
		}

		uint64_t cursor =
			static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
		for (size_t context_id = 0;
		     context_id < context_sq_sizes.size(); context_id++) {
			context_slice_bases[context_id] = cursor;
			cursor +=
				static_cast<uint64_t>(context_sq_sizes[context_id]) *
				slot_size_bytes;
		}
	} catch (...) {
		release();
		throw;
	}
}

uint32_t gdaki_putvalue_v1_staging::lkey_for_rail(uint16_t rail_id) const
{
	if (rail_id >= rail_registrations.size() ||
	    rail_registrations[rail_id].mr == nullptr) {
		throw std::runtime_error(
			"gdaki_putvalue_v1_staging: rail is not registered");
	}
	return rail_registrations[rail_id].lkey;
}

uint64_t gdaki_putvalue_v1_staging::slice_base_for_context(
	size_t context_id) const
{
	if (context_id >= context_slice_bases.size()) {
		throw std::runtime_error(
			"gdaki_putvalue_v1_staging: context has no slice");
	}
	return context_slice_bases[context_id];
}

void gdaki_sc_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     struct fi_efa_ops_gda *gda_ops, int backend_version)
{
	/* Create hardware counters first; they will be bound to the inner
	 * endpoint between open() and enable(). */
	write_cntr.create(gda_ops, domain);
	remote_write_cntr.create(gda_ops, domain);

	/* Open the inner endpoint without enable. Use the same CQ sizing as
	 * the data endpoint so callers get consistent capacity per env config. */
	base.endpoint.open(domain, ref_info, ofi_nccl_cq_size(), backend_version);

	/* Bind counters before enabling. */
	base.endpoint.bind(&write_cntr.get()->fid, FI_WRITE);
	base.endpoint.bind(&remote_write_cntr.get()->fid, FI_REMOTE_WRITE);

	base.endpoint.enable();
}

void gdaki_sc_endpoint::populate(int backend_version, struct fi_efa_ops_gda *gda_ops,
				 const std::vector<uint8_t> &all_addrs,
				 size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP/CQ query, MMIO map, GPU descriptors,
	 * target table) to the inner endpoint. */
	base.populate(backend_version, gda_ops, all_addrs, ep_addr_len, total_slots, nranks);

	/*
	 * Build the two device handles. They share QP / CQ / target
	 * addressing / sq_lock / sq_size / submitted_count layout — only
	 * the (cntr_value, local_cntr_value) pair differs.
	 *
	 * - counter_dev_handle exposes the WRITE counter via cntr_value
	 *   (FI_WRITE — local completion). Returned to the kernel through
	 *   counter_handles[].
	 * - signal_dev_handle exposes the REMOTE_WRITE counter via cntr_value
	 *   (FI_REMOTE_WRITE — signal arrival), and the WRITE counter via
	 *   local_cntr_value (used by the device for backpressure / Flush).
	 *   Returned to the kernel through signal_handles[].
	 *
	 * Both `cntr_value` and `local_cntr_value` are set on the host before
	 * commit() pushes the struct to GPU memory.
	 */
	auto fill_common = [&](nccl_ofi_gin_gdaki_dev_counter_handle &h) {
		h.base.qp = base.gpu_qp.dev();
		h.base.cq = base.gpu_cq.dev();
		h.base.target_address_handles = base.targets.ahs.dev;
		h.base.target_remote_qpns = base.targets.qpns.dev;
		h.base.target_qkey = base.targets.qkeys.dev;
		h.base.sq_lock = 0;
		h.base.submitted_count = 0;
		h.base.sq_size = base.sq_size;
		h.base.putvalue_pad = 0;
		h.base.putvalue_slice_base = 0;
		h.cntr_offset = 0;   /* offset-based reset baseline */
	};

	counter_dev_handle.allocate(1);
	fill_common(counter_dev_handle.host[0]);
	/* TODO: Refactor counter_dev_handle so the same gpu memory is not
	 * being used by multiple fields */
	counter_dev_handle.host[0].cntr_value = write_cntr.gpu_ptr();
	counter_dev_handle.host[0].base.local_cntr_value = write_cntr.gpu_ptr();
	counter_dev_handle.commit();

	signal_dev_handle.allocate(1);
	fill_common(signal_dev_handle.host[0]);
	signal_dev_handle.host[0].cntr_value = remote_write_cntr.gpu_ptr();
	signal_dev_handle.host[0].base.local_cntr_value = write_cntr.gpu_ptr();
	signal_dev_handle.commit();
}
