/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin for the GIN API. Shared APIs (init, devices, listen, connect,
 * regMrSym[DmaBuf], deregMrSym, closeColl, closeListen, ginProgress, finalize)
 * are reused from the proxy-side implementations in nccl_ofi_gin_api.cpp.
 * Only the GDAKI-specific APIs (createContext/destroyContext/get_properties/
 * queryLastError) live here.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

#if HAVE_DECL_FI_EFA_GDA_OPS
#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_ctx.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_ofiutils.h"

#include <algorithm>

#define GDAKI_ENABLED 1
#else
#define GDAKI_ENABLED 0
#endif /* HAVE_DECL_FI_EFA_GDA_OPS */

bool nccl_ofi_gin_gdaki_enabled()
{
	return ofi_nccl_gin_gdaki.get();
}

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
	props->maxRecvs = ofi_properties.max_group_receives;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDAKI;
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

#if GDAKI_ENABLED

/*
 * Helper: get fi_info for the efa-direct fabric. Takes the domain name from
 * the existing proxy fi_info (so we open the same EFA device) but requests
 * fabric "efa-direct" instead of "efa".
 */
static struct fi_info *get_gdaki_info(const char *domain_name)
{
	ofi_info_ptr hints(fi_allocinfo());
	if (!hints) {
		throw std::runtime_error("fi_allocinfo failed");
	}

	hints->caps = FI_MSG | FI_RMA | FI_WRITE | FI_READ | FI_SEND | FI_RECV | FI_SOURCE;
	hints->mode = FI_CONTEXT2;
	hints->ep_attr->type = FI_EP_RDM;

	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
	hints->domain_attr->threading = FI_THREAD_SAFE;
	hints->domain_attr->data_progress = FI_PROGRESS_AUTO;
	hints->domain_attr->control_progress = FI_PROGRESS_AUTO;

	hints->domain_attr->name = strdup(domain_name);
	hints->fabric_attr->name = strdup("efa-direct");

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(2, 0), nullptr, nullptr, 0ULL, hints.get(), &results);
	if (ret != 0) {
		NCCL_OFI_WARN("fi_getinfo for efa-direct failed: %s", fi_strerror(-ret));
		throw std::runtime_error("fi_getinfo for efa-direct failed");
	}
	if (results == nullptr) {
		throw std::runtime_error("No efa-direct provider available");
	}

	return results;
}

/*
 * Helper: GPU-allocate and zero a buffer using plugin CUDA wrappers.
 */
static void *gpu_alloc_zeroed(size_t size)
{
	void *ptr = nullptr;
	if (nccl_net_ofi_gpu_mem_alloc(&ptr, size) != 0) {
		return nullptr;
	}
	/* cuMemAlloc doesn't zero — use memcpy of a zeroed host buffer */
	void *zeros = calloc(1, size);
	if (!zeros) {
		nccl_net_ofi_gpu_mem_free(ptr);
		return nullptr;
	}
	if (nccl_net_ofi_gpu_mem_copy_host_to_device(ptr, zeros, size) != 0) {
		free(zeros);
		nccl_net_ofi_gpu_mem_free(ptr);
		return nullptr;
	}
	free(zeros);
	return ptr;
}

static void gpu_free(void *ptr)
{
	if (ptr) {
		nccl_net_ofi_gpu_mem_free(ptr);
	}
}

#if HAVE_FI_EFA_COMP_CNTR
/*
 * Create a hardware counter via cntr_open_ext with GPU-accessible external
 * memory for the counter value. Returns the fi_cntr and the GPU memory
 * pointer. The counter value is zero-initialized.
 */
static struct fid_cntr *create_hw_counter(struct fi_efa_ops_gda *gda_ops,
					  struct fid_domain *domain,
					  void **d_cntr_mem_out)
{
	/* Allocate GPU memory for the 64-bit counter value */
	void *d_mem = gpu_alloc_zeroed(sizeof(uint64_t));
	if (!d_mem) {
		throw std::runtime_error("gpu_alloc for counter memory failed");
	}

	struct fi_cntr_attr cntr_attr = {};
	cntr_attr.events = FI_CNTR_EVENTS_COMP;

	struct fi_efa_comp_cntr_init_attr efa_attr = {};
	efa_attr.flags = FI_EFA_COMP_CNTR_INIT_WITH_COMP_EXTERNAL_MEM;
	efa_attr.comp_cntr_ext_mem.ptr = static_cast<uint8_t *>(d_mem);
	efa_attr.comp_cntr_ext_mem.type = FI_EFA_MEMORY_LOCATION_VA;

	struct fid_cntr *cntr = nullptr;
	int ret = gda_ops->cntr_open_ext(domain, &cntr_attr, &cntr, nullptr, &efa_attr);
	if (ret != 0) {
		gpu_free(d_mem);
		throw std::runtime_error("cntr_open_ext failed: " +
					 std::string(fi_strerror(-ret)));
	}

	*d_cntr_mem_out = d_mem;
	return cntr;
}

/*
 * Create one signal/counter endpoint: EP + CQ + two hardware counters
 * (FI_WRITE and FI_REMOTE_WRITE), bind them, enable, query QP/CQ attrs,
 * register SQ MMIO for GPU access, build QP/CQ structs, upload to GPU.
 */
static void create_sc_endpoint(nccl_ofi_gin_gdaki_sc_endpoint *sc,
			       struct fi_info *info,
			       struct fid_domain *domain,
			       struct fid_av *av,
			       struct fi_efa_ops_gda *gda_ops)
{
	memset(sc, 0, sizeof(*sc));
	int ret;

	/* CQ */
	struct fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = 128;
	ret = fi_cq_open(domain, &cq_attr, &sc->cq, nullptr);
	if (ret != 0)
		throw std::runtime_error("fi_cq_open for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	/* Hardware counters with GPU-accessible external memory */
	sc->write_cntr = create_hw_counter(gda_ops, domain, &sc->d_write_cntr_mem);
	sc->remote_write_cntr = create_hw_counter(gda_ops, domain, &sc->d_remote_write_cntr_mem);

	/* EP */
	ret = fi_endpoint(domain, info, &sc->ep, nullptr);
	if (ret != 0)
		throw std::runtime_error("fi_endpoint for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	ret = fi_ep_bind(sc->ep, &sc->cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret != 0)
		throw std::runtime_error("fi_ep_bind CQ for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	ret = fi_ep_bind(sc->ep, &av->fid, 0);
	if (ret != 0)
		throw std::runtime_error("fi_ep_bind AV for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	ret = fi_ep_bind(sc->ep, &sc->write_cntr->fid, FI_WRITE);
	if (ret != 0)
		throw std::runtime_error("fi_ep_bind FI_WRITE counter failed: " +
					 std::string(fi_strerror(-ret)));

	ret = fi_ep_bind(sc->ep, &sc->remote_write_cntr->fid, FI_REMOTE_WRITE);
	if (ret != 0)
		throw std::runtime_error("fi_ep_bind FI_REMOTE_WRITE counter failed: " +
					 std::string(fi_strerror(-ret)));

	ret = fi_enable(sc->ep);
	if (ret != 0)
		throw std::runtime_error("fi_enable for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	/* Query QP work queues and build GPU-resident QP struct */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	ret = gda_ops->query_qp_wqs(sc->ep, &sq_attr, &rq_attr);
	if (ret != 0)
		throw std::runtime_error("query_qp_wqs for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	/* Register SQ MMIO for GPU access */
	void *d_sq_buf = nullptr, *d_sq_db = nullptr;
	if (nccl_net_ofi_gpu_host_register_iomem(sq_attr.buffer,
			(size_t)sq_attr.num_entries * sq_attr.entry_size) != 0)
		throw std::runtime_error("host_register SQ buffer for sc endpoint failed");
	if (nccl_net_ofi_gpu_host_get_device_pointer(&d_sq_buf, sq_attr.buffer) != 0)
		throw std::runtime_error("get_device_pointer SQ buffer for sc endpoint failed");
	if (nccl_net_ofi_gpu_host_register_iomem(sq_attr.doorbell, 4096) != 0)
		throw std::runtime_error("host_register SQ doorbell for sc endpoint failed");
	if (nccl_net_ofi_gpu_host_get_device_pointer(&d_sq_db, sq_attr.doorbell) != 0)
		throw std::runtime_error("get_device_pointer SQ doorbell for sc endpoint failed");

	nccl_ofi_gin_gdaki_qp h_qp = {};
	h_qp.sq.wq.buf = (uint8_t *)d_sq_buf;
	h_qp.sq.wq.db = (uint32_t *)d_sq_db;
	h_qp.sq.wq.max_wqes = sq_attr.num_entries;
	h_qp.sq.wq.queue_mask = sq_attr.num_entries - 1;
	h_qp.sq.wq.queue_size_shift = __builtin_ctz(sq_attr.num_entries);
	h_qp.sq.wq.max_batch = sq_attr.max_batch;
	h_qp.sq.wq.phase = 0;
	h_qp.sq.max_inline_data = 32;
	h_qp.sq.max_rdma_sges = 2;
	h_qp.rq.wq.buf = rq_attr.buffer;
	h_qp.rq.wq.db = rq_attr.doorbell;
	h_qp.rq.wq.max_wqes = rq_attr.num_entries;
	h_qp.rq.wq.queue_mask = rq_attr.num_entries - 1;
	h_qp.rq.wq.queue_size_shift = __builtin_ctz(rq_attr.num_entries);
	h_qp.rq.wq.max_batch = rq_attr.num_entries;
	h_qp.rq.wq.phase = 1;

	sc->d_qp = gpu_alloc_zeroed(sizeof(h_qp));
	if (!sc->d_qp)
		throw std::runtime_error("gpu_alloc QP for sc endpoint failed");
	if (nccl_net_ofi_gpu_mem_copy_host_to_device(sc->d_qp, &h_qp, sizeof(h_qp)) != 0)
		throw std::runtime_error("gpu_mem_copy QP for sc endpoint failed");

	/* Query CQ and build GPU-resident CQ struct */
	struct fi_efa_cq_attr efa_cq_attr = {};
	ret = gda_ops->query_cq(sc->cq, &efa_cq_attr);
	if (ret != 0)
		throw std::runtime_error("query_cq for sc endpoint failed: " +
					 std::string(fi_strerror(-ret)));

	nccl_ofi_gin_gdaki_cq h_cq = {};
	h_cq.buf = efa_cq_attr.buffer;
	h_cq.entry_size = efa_cq_attr.entry_size;
	h_cq.num_entries = efa_cq_attr.num_entries;
	h_cq.queue_mask = efa_cq_attr.num_entries - 1;
	h_cq.queue_size_shift = __builtin_ctz(efa_cq_attr.num_entries);
	h_cq.phase = 1;

	sc->d_cq = gpu_alloc_zeroed(sizeof(h_cq));
	if (!sc->d_cq)
		throw std::runtime_error("gpu_alloc CQ for sc endpoint failed");
	if (nccl_net_ofi_gpu_mem_copy_host_to_device(sc->d_cq, &h_cq, sizeof(h_cq)) != 0)
		throw std::runtime_error("gpu_mem_copy CQ for sc endpoint failed");
}
#endif /* HAVE_FI_EFA_COMP_CNTR */

/*
 * Tear down a single signal/counter endpoint. Safe on partial init.
 */
static void gdaki_destroy_sc_endpoint(nccl_ofi_gin_gdaki_sc_endpoint *sc)
{
	if (!sc) return;

	gpu_free(sc->d_dev_handle);
	gpu_free(sc->d_address_handles);
	gpu_free(sc->d_remote_qpns);
	gpu_free(sc->d_qkey);
	gpu_free(sc->d_qp);
	gpu_free(sc->d_cq);
	gpu_free(sc->d_write_cntr_mem);
	gpu_free(sc->d_remote_write_cntr_mem);

	if (sc->ep) fi_close(&sc->ep->fid);
	if (sc->cq) fi_close(&sc->cq->fid);
	if (sc->write_cntr) fi_close(&sc->write_cntr->fid);
	if (sc->remote_write_cntr) fi_close(&sc->remote_write_cntr->fid);
}

/*
 * Tear down everything in ctx. Safe to call on a partially-initialized ctx.
 */
static void gdaki_destroy_ctx(nccl_ofi_gin_gdaki_context *ctx)
{
	if (!ctx) {
		return;
	}

	/* 1. Free signal/counter endpoints */
	for (int i = 0; i < ctx->n_sc_endpoints; i++) {
		gdaki_destroy_sc_endpoint(&ctx->sc_endpoints[i]);
	}
	delete[] ctx->sc_endpoints;
	gpu_free(ctx->d_signal_handles);
	gpu_free(ctx->d_counter_handles);

	/* 2. Free per-peer GPU arrays */
	gpu_free(ctx->address_handles_dev);
	gpu_free(ctx->remote_qpns_dev);
	gpu_free(ctx->qkey_dev);

	/* 3. Free QP and CQ GPU allocations */
	gpu_free(ctx->d_qp);
	gpu_free(ctx->d_cq);

	/* 4. Free device handle struct */
	gpu_free(ctx->d_handle);

	/* 5. Deregister and free SQ/CQ GPU buffers */
	if (ctx->sq_mr) {
		fi_close(&ctx->sq_mr->fid);
	}
	gpu_free(ctx->sq_buffer_dev);
	if (ctx->cq_mr) {
		fi_close(&ctx->cq_mr->fid);
	}
	gpu_free(ctx->cq_buffer_dev);

	/* 6. Close libfabric resources (order matters: ep before cq/av/domain/fabric) */
	if (ctx->ofi_ep) {
		fi_close(&ctx->ofi_ep->fid);
	}
	if (ctx->ofi_cq) {
		fi_close(&ctx->ofi_cq->fid);
	}
	if (ctx->ofi_av) {
		fi_close(&ctx->ofi_av->fid);
	}
	if (ctx->ofi_domain) {
		fi_close(&ctx->ofi_domain->fid);
	}
	if (ctx->ofi_fabric) {
		fi_close(&ctx->ofi_fabric->fid);
	}
	if (ctx->ofi_info) {
		fi_freeinfo(ctx->ofi_info);
	}

	delete ctx;
}

#endif /* GDAKI_ENABLED */

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
#if !GDAKI_ENABLED
	(void)collComm;
	(void)config;
	(void)ginCtx;
	(void)devHandle;
	NCCL_OFI_WARN("gin GDAKI: createContext requires efa-dp-direct and FI_EFA_GDA_OPS; "
		      "plugin was built without one or both");
	return ncclInternalError;
#else
	if (collComm == nullptr || config == nullptr || ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	int nranks = put_comm->get_nranks();
	int rank = put_comm->get_rank();

	auto *ctx = new (std::nothrow) nccl_ofi_gin_gdaki_context();
	if (ctx == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed to allocate context");
		return ncclSystemError;
	}
	/* Zero-init so gdaki_destroy_ctx is safe on partial init */
	memset(ctx, 0, sizeof(*ctx));
	ctx->nSignals = config->nSignals;
	ctx->nCounters = config->nCounters;
	ctx->nranks = nranks;
	ctx->rank = rank;

	try {
		/*
		 * Step 1: Get efa-direct fi_info. We need the domain name from
		 * the existing proxy transport to target the same EFA device.
		 */
		auto *plugin = nccl_net_ofi_get_plugin();
		auto *device = plugin->get_device(put_comm->get_dev());
		if (device == nullptr) {
			throw std::runtime_error("get_device returned null");
		}
		struct fi_info *proxy_info = device->get_ofi_info(0);

		ctx->ofi_info = get_gdaki_info(proxy_info->domain_attr->name);

		/*
		 * Step 2: Open fabric and domain on the efa-direct provider.
		 */
		int ret;
		ret = fi_fabric(ctx->ofi_info->fabric_attr, &ctx->ofi_fabric, nullptr);
		if (ret != 0) {
			throw std::runtime_error("fi_fabric for efa-direct failed: " +
						 std::string(fi_strerror(-ret)));
		}

		ret = fi_domain(ctx->ofi_fabric, ctx->ofi_info, &ctx->ofi_domain, nullptr);
		if (ret != 0) {
			throw std::runtime_error("fi_domain for efa-direct failed: " +
						 std::string(fi_strerror(-ret)));
		}

		/*
		 * Step 3: Create CQ and AV.
		 */
		struct fi_cq_attr cq_attr = {};
		cq_attr.format = FI_CQ_FORMAT_DATA;
		cq_attr.size = 1024; /* TODO: tune based on queueDepth */
		ret = fi_cq_open(ctx->ofi_domain, &cq_attr, &ctx->ofi_cq, nullptr);
		if (ret != 0) {
			throw std::runtime_error("fi_cq_open for efa-direct failed: " +
						 std::string(fi_strerror(-ret)));
		}

		struct fi_av_attr av_attr = {};
		av_attr.type = FI_AV_TABLE;
		ret = fi_av_open(ctx->ofi_domain, &av_attr, &ctx->ofi_av, nullptr);
		if (ret != 0) {
			throw std::runtime_error("fi_av_open for efa-direct failed: " +
						 std::string(fi_strerror(-ret)));
		}

		/*
		 * Step 4: Create endpoint, bind CQ and AV, enable.
		 */
		ret = fi_endpoint(ctx->ofi_domain, ctx->ofi_info, &ctx->ofi_ep, nullptr);
		if (ret != 0) {
			throw std::runtime_error("fi_endpoint for efa-direct failed: " +
						 std::string(fi_strerror(-ret)));
		}

		ret = fi_ep_bind(ctx->ofi_ep, &ctx->ofi_cq->fid, FI_TRANSMIT | FI_RECV);
		if (ret != 0) {
			throw std::runtime_error("fi_ep_bind CQ failed: " +
						 std::string(fi_strerror(-ret)));
		}

		ret = fi_ep_bind(ctx->ofi_ep, &ctx->ofi_av->fid, 0);
		if (ret != 0) {
			throw std::runtime_error("fi_ep_bind AV failed: " +
						 std::string(fi_strerror(-ret)));
		}

		ret = fi_enable(ctx->ofi_ep);
		if (ret != 0) {
			throw std::runtime_error("fi_enable for efa-direct failed: " +
						 std::string(fi_strerror(-ret)));
		}

		/*
		 * Step 5: Open GDA ops extension.
		 */
		struct fi_efa_ops_gda *gda_ops = nullptr;
		ret = fi_open_ops(&ctx->ofi_domain->fid, FI_EFA_GDA_OPS, 0,
				  (void **)&gda_ops, nullptr);
		if (ret != 0 || gda_ops == nullptr) {
			throw std::runtime_error("fi_open_ops FI_EFA_GDA_OPS failed: " +
						 std::string(fi_strerror(-ret)));
		}

		/*
		 * Step 6: Query QP work queues and CQ attributes.
		 */
		struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
		ret = gda_ops->query_qp_wqs(ctx->ofi_ep, &sq_attr, &rq_attr);
		if (ret != 0) {
			throw std::runtime_error("query_qp_wqs failed: " +
						 std::string(fi_strerror(-ret)));
		}

		struct fi_efa_cq_attr efa_cq_attr = {};
		ret = gda_ops->query_cq(ctx->ofi_cq, &efa_cq_attr);
		if (ret != 0) {
			throw std::runtime_error("query_cq failed: " +
						 std::string(fi_strerror(-ret)));
		}

		/*
		 * Step 7: Register SQ MMIO regions for GPU access.
		 * The SQ buffer and doorbell are device MMIO (BAR) regions that
		 * require cuMemHostRegister with IOMEMORY | DEVICEMAP for GPU
		 * kernels to write WQEs and ring the doorbell.
		 *
		 * TODO: RQ buffer registration as DEVICEMAP works.
		 * Revisit when RDMA_READ is implemented.
		 *
		 * TODO: CQ buffer registration fails with both IOMEMORY and
		 * DEVICEMAP on the current p5en cluster. The host pointer
		 * works for both CPU and GPU CQ polling.
		 */
		void *d_sq_buf = nullptr, *d_sq_db = nullptr;

		if (nccl_net_ofi_gpu_host_register_iomem(sq_attr.buffer,
				(size_t)sq_attr.num_entries * sq_attr.entry_size) != 0)
			throw std::runtime_error("host_register SQ buffer failed");
		if (nccl_net_ofi_gpu_host_get_device_pointer(&d_sq_buf, sq_attr.buffer) != 0)
			throw std::runtime_error("get_device_pointer SQ buffer failed");

		if (nccl_net_ofi_gpu_host_register_iomem(sq_attr.doorbell, 4096) != 0)
			throw std::runtime_error("host_register SQ doorbell failed");
		if (nccl_net_ofi_gpu_host_get_device_pointer(&d_sq_db, sq_attr.doorbell) != 0)
			throw std::runtime_error("get_device_pointer SQ doorbell failed");

		/* Build QP struct (layout-compatible with efa_cuda_qp) */
		nccl_ofi_gin_gdaki_qp h_qp = {};
		h_qp.sq.wq.buf = (uint8_t *)d_sq_buf;
		h_qp.sq.wq.db = (uint32_t *)d_sq_db;
		h_qp.sq.wq.max_wqes = sq_attr.num_entries;
		h_qp.sq.wq.queue_mask = sq_attr.num_entries - 1;
		h_qp.sq.wq.queue_size_shift = __builtin_ctz(sq_attr.num_entries);
		h_qp.sq.wq.max_batch = sq_attr.max_batch;
		h_qp.sq.wq.phase = 0;
		h_qp.sq.max_inline_data = 32;
		h_qp.sq.max_rdma_sges = 2;
		h_qp.rq.wq.buf = rq_attr.buffer;
		h_qp.rq.wq.db = rq_attr.doorbell;
		h_qp.rq.wq.max_wqes = rq_attr.num_entries;
		h_qp.rq.wq.queue_mask = rq_attr.num_entries - 1;
		h_qp.rq.wq.queue_size_shift = __builtin_ctz(rq_attr.num_entries);
		h_qp.rq.wq.max_batch = rq_attr.num_entries;
		h_qp.rq.wq.phase = 1;

		/* Build CQ struct (layout-compatible with efa_cuda_cq) */
		nccl_ofi_gin_gdaki_cq h_cq = {};
		h_cq.buf = efa_cq_attr.buffer;
		h_cq.entry_size = efa_cq_attr.entry_size;
		h_cq.num_entries = efa_cq_attr.num_entries;
		h_cq.queue_mask = efa_cq_attr.num_entries - 1;
		h_cq.queue_size_shift = __builtin_ctz(efa_cq_attr.num_entries);
		h_cq.phase = 1;

		/* Upload QP and CQ to GPU */
		nccl_ofi_gin_gdaki_qp *d_qp = nullptr;
		if (nccl_net_ofi_gpu_mem_alloc((void **)&d_qp, sizeof(h_qp)) != 0)
			throw std::runtime_error("gpu_mem_alloc QP failed");
		if (nccl_net_ofi_gpu_mem_copy_host_to_device(d_qp, &h_qp, sizeof(h_qp)) != 0)
			throw std::runtime_error("gpu_mem_copy QP failed");

		nccl_ofi_gin_gdaki_cq *d_cq = nullptr;
		if (nccl_net_ofi_gpu_mem_alloc((void **)&d_cq, sizeof(h_cq)) != 0)
			throw std::runtime_error("gpu_mem_alloc CQ failed");
		if (nccl_net_ofi_gpu_mem_copy_host_to_device(d_cq, &h_cq, sizeof(h_cq)) != 0)
			throw std::runtime_error("gpu_mem_copy CQ failed");

		/* Store in ctx for teardown */
		ctx->d_qp = d_qp;
		ctx->d_cq = d_cq;

		/*
		 * Step 8: Allgather endpoint addresses.
		 */
		size_t ep_addr_len = 0;
		fi_getname(&ctx->ofi_ep->fid, nullptr, &ep_addr_len);
		/* ep_addr_len now holds the required size */

		std::vector<uint8_t> all_addrs(nranks * ep_addr_len, 0);
		ret = fi_getname(&ctx->ofi_ep->fid, &all_addrs[rank * ep_addr_len], &ep_addr_len);
		if (ret != 0) {
			throw std::runtime_error("fi_getname failed: " +
						 std::string(fi_strerror(-ret)));
		}

		ret = put_comm->get_ag_comm().all_gather(all_addrs.data(), ep_addr_len);
		if (ret != 0) {
			throw std::runtime_error("allgather of ep addresses failed");
		}

		/*
		 * Step 9: Insert peers into AV and query per-peer addressing.
		 */
		std::vector<uint16_t> h_ahns(nranks);
		std::vector<uint16_t> h_qpns(nranks);
		std::vector<uint32_t> h_qkeys(nranks);

		for (int i = 0; i < nranks; i++) {
			fi_addr_t fi_addr;
			ret = fi_av_insert(ctx->ofi_av, &all_addrs[i * ep_addr_len], 1,
					   &fi_addr, 0, nullptr);
			if (ret != 1) {
				throw std::runtime_error("fi_av_insert failed for rank " +
							 std::to_string(i));
			}

			uint16_t ahn = 0;
			uint16_t remote_qpn = 0;
			uint32_t remote_qkey = 0;
			ret = gda_ops->query_addr(ctx->ofi_ep, fi_addr, &ahn,
						  &remote_qpn, &remote_qkey);
			if (ret != 0) {
				throw std::runtime_error("query_addr failed for rank " +
							 std::to_string(i));
			}

			h_ahns[i] = ahn;
			h_qpns[i] = remote_qpn;
			h_qkeys[i] = remote_qkey;
		}

		/*
		 * Step 10: Allocate GPU memory for per-peer arrays and copy.
		 */
		size_t ahn_size = nranks * sizeof(uint16_t);
		size_t qpn_size = nranks * sizeof(uint16_t);
		size_t qkey_size = nranks * sizeof(uint32_t);

		ctx->address_handles_dev = static_cast<uint16_t *>(gpu_alloc_zeroed(ahn_size));
		ctx->remote_qpns_dev = static_cast<uint16_t *>(gpu_alloc_zeroed(qpn_size));
		ctx->qkey_dev = static_cast<uint32_t *>(gpu_alloc_zeroed(qkey_size));
		if (!ctx->address_handles_dev || !ctx->remote_qpns_dev || !ctx->qkey_dev) {
			throw std::runtime_error("gpu_alloc for per-peer arrays failed");
		}

		int mc_ret;
		mc_ret = nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->address_handles_dev, h_ahns.data(), ahn_size);
		if (mc_ret) throw std::runtime_error("memcpy ahns failed");
		mc_ret = nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->remote_qpns_dev, h_qpns.data(), qpn_size);
		if (mc_ret) throw std::runtime_error("memcpy qpns failed");
		mc_ret = nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->qkey_dev, h_qkeys.data(), qkey_size);
		if (mc_ret) throw std::runtime_error("memcpy qkeys failed");

		/*
		 * Step 11: Create signal/counter endpoints if requested.
		 */
#if HAVE_FI_EFA_COMP_CNTR
		int n_sc = std::max(config->nSignals, config->nCounters);
		if (n_sc > 0) {
			ctx->sc_endpoints = new (std::nothrow) nccl_ofi_gin_gdaki_sc_endpoint[n_sc]();
			if (!ctx->sc_endpoints)
				throw std::runtime_error("alloc sc_endpoints array failed");
			ctx->n_sc_endpoints = n_sc;

			for (int i = 0; i < n_sc; i++) {
				create_sc_endpoint(&ctx->sc_endpoints[i], ctx->ofi_info,
						   ctx->ofi_domain, ctx->ofi_av, gda_ops);
			}

			/*
			 * Allgather signal/counter endpoint addresses and
			 * populate per-peer addressing arrays.
			 */
			for (int i = 0; i < n_sc; i++) {
				auto *sc = &ctx->sc_endpoints[i];

				size_t sc_addr_len = 0;
				fi_getname(&sc->ep->fid, nullptr, &sc_addr_len);

				std::vector<uint8_t> sc_all_addrs(nranks * sc_addr_len, 0);
				ret = fi_getname(&sc->ep->fid, &sc_all_addrs[rank * sc_addr_len], &sc_addr_len);
				if (ret != 0)
					throw std::runtime_error("fi_getname for sc endpoint failed");

				ret = put_comm->get_ag_comm().all_gather(sc_all_addrs.data(), sc_addr_len);
				if (ret != 0)
					throw std::runtime_error("allgather of sc endpoint addresses failed");

				std::vector<uint16_t> sc_ahns(nranks);
				std::vector<uint16_t> sc_qpns(nranks);
				std::vector<uint32_t> sc_qkeys(nranks);

				for (int j = 0; j < nranks; j++) {
					fi_addr_t fi_addr;
					ret = fi_av_insert(ctx->ofi_av, &sc_all_addrs[j * sc_addr_len],
							   1, &fi_addr, 0, nullptr);
					if (ret != 1)
						throw std::runtime_error("fi_av_insert for sc endpoint failed");

					uint16_t ahn = 0;
					uint16_t remote_qpn = 0;
					uint32_t remote_qkey = 0;
					ret = gda_ops->query_addr(sc->ep, fi_addr, &ahn,
								  &remote_qpn, &remote_qkey);
					if (ret != 0)
						throw std::runtime_error("query_addr for sc endpoint failed");

					sc_ahns[j] = ahn;
					sc_qpns[j] = remote_qpn;
					sc_qkeys[j] = remote_qkey;
				}

				/* Upload per-peer arrays to GPU */
				sc->d_address_handles = static_cast<uint16_t *>(gpu_alloc_zeroed(ahn_size));
				sc->d_remote_qpns = static_cast<uint16_t *>(gpu_alloc_zeroed(qpn_size));
				sc->d_qkey = static_cast<uint32_t *>(gpu_alloc_zeroed(qkey_size));
				if (!sc->d_address_handles || !sc->d_remote_qpns || !sc->d_qkey)
					throw std::runtime_error("gpu_alloc for sc per-peer arrays failed");

				if (nccl_net_ofi_gpu_mem_copy_host_to_device(sc->d_address_handles, sc_ahns.data(), ahn_size) != 0)
					throw std::runtime_error("memcpy sc ahns failed");
				if (nccl_net_ofi_gpu_mem_copy_host_to_device(sc->d_remote_qpns, sc_qpns.data(), qpn_size) != 0)
					throw std::runtime_error("memcpy sc qpns failed");
				if (nccl_net_ofi_gpu_mem_copy_host_to_device(sc->d_qkey, sc_qkeys.data(), qkey_size) != 0)
					throw std::runtime_error("memcpy sc qkeys failed");

				/* Build and upload the per-endpoint device handle */
				nccl_ofi_gin_dev_counter_handle h_dev = {};
				h_dev.qp = static_cast<nccl_ofi_gin_gdaki_qp *>(sc->d_qp);
				h_dev.cq = static_cast<nccl_ofi_gin_gdaki_cq *>(sc->d_cq);
				h_dev.cntr_value = nullptr; /* set below per signal/counter */
				h_dev.address_handles = sc->d_address_handles;
				h_dev.remote_qpns = sc->d_remote_qpns;
				h_dev.qkey = sc->d_qkey;

				sc->d_dev_handle = static_cast<nccl_ofi_gin_dev_counter_handle *>(
					gpu_alloc_zeroed(sizeof(h_dev)));
				if (!sc->d_dev_handle)
					throw std::runtime_error("gpu_alloc for sc dev handle failed");
				if (nccl_net_ofi_gpu_mem_copy_host_to_device(sc->d_dev_handle, &h_dev, sizeof(h_dev)) != 0)
					throw std::runtime_error("gpu_mem_copy sc dev handle failed");
			}

			/*
			 * Build GPU arrays of device handle pointers for
			 * signal_handles[] and counter_handles[] in the
			 * main device handle.
			 *
			 * For signals: cntr_value points to the FI_REMOTE_WRITE counter.
			 * For counters: cntr_value points to the FI_WRITE counter.
			 * We patch the cntr_value in the already-uploaded device handles.
			 */
			if (config->nSignals > 0) {
				std::vector<nccl_ofi_gin_dev_counter_handle *> h_sig_ptrs(config->nSignals);
				for (int i = 0; i < config->nSignals; i++) {
					h_sig_ptrs[i] = ctx->sc_endpoints[i].d_dev_handle;
					/* Patch cntr_value to point to FI_REMOTE_WRITE counter mem */
					volatile uint64_t *cntr_ptr = static_cast<volatile uint64_t *>(
						ctx->sc_endpoints[i].d_remote_write_cntr_mem);
					size_t offset = offsetof(nccl_ofi_gin_dev_counter_handle, cntr_value);
					if (nccl_net_ofi_gpu_mem_copy_host_to_device(
						    reinterpret_cast<uint8_t *>(ctx->sc_endpoints[i].d_dev_handle) + offset,
						    &cntr_ptr, sizeof(cntr_ptr)) != 0)
						throw std::runtime_error("patch signal cntr_value failed");
				}
				size_t sig_arr_size = config->nSignals * sizeof(nccl_ofi_gin_dev_counter_handle *);
				ctx->d_signal_handles = static_cast<nccl_ofi_gin_dev_counter_handle **>(
					gpu_alloc_zeroed(sig_arr_size));
				if (!ctx->d_signal_handles)
					throw std::runtime_error("gpu_alloc signal_handles array failed");
				if (nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->d_signal_handles, h_sig_ptrs.data(), sig_arr_size) != 0)
					throw std::runtime_error("memcpy signal_handles failed");
			}

			if (config->nCounters > 0) {
				std::vector<nccl_ofi_gin_dev_counter_handle *> h_ctr_ptrs(config->nCounters);
				for (int i = 0; i < config->nCounters; i++) {
					h_ctr_ptrs[i] = ctx->sc_endpoints[i].d_dev_handle;
					/* Patch cntr_value to point to FI_WRITE counter mem */
					volatile uint64_t *cntr_ptr = static_cast<volatile uint64_t *>(
						ctx->sc_endpoints[i].d_write_cntr_mem);
					size_t offset = offsetof(nccl_ofi_gin_dev_counter_handle, cntr_value);
					if (nccl_net_ofi_gpu_mem_copy_host_to_device(
						    reinterpret_cast<uint8_t *>(ctx->sc_endpoints[i].d_dev_handle) + offset,
						    &cntr_ptr, sizeof(cntr_ptr)) != 0)
						throw std::runtime_error("patch counter cntr_value failed");
				}
				size_t ctr_arr_size = config->nCounters * sizeof(nccl_ofi_gin_dev_counter_handle *);
				ctx->d_counter_handles = static_cast<nccl_ofi_gin_dev_counter_handle **>(
					gpu_alloc_zeroed(ctr_arr_size));
				if (!ctx->d_counter_handles)
					throw std::runtime_error("gpu_alloc counter_handles array failed");
				if (nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->d_counter_handles, h_ctr_ptrs.data(), ctr_arr_size) != 0)
					throw std::runtime_error("memcpy counter_handles failed");
			}
		}
#else
		if (config->nSignals > 0 || config->nCounters > 0) {
			NCCL_OFI_WARN("gin GDAKI: signals/counters requested but libfabric lacks "
				      "cntr_open_ext support (fi_efa_comp_cntr_init_attr not found)");
			throw std::runtime_error("signals/counters not supported by this libfabric");
		}
#endif /* HAVE_FI_EFA_COMP_CNTR */

		/*
		 * Step 12: Populate and upload the device handle.
		 */
		nccl_ofi_gin_gdaki_dev_handle h_handle = {};
		h_handle.qp = d_qp;
		h_handle.cq = d_cq;
		h_handle.address_handles = ctx->address_handles_dev;
		h_handle.remote_qpns = ctx->remote_qpns_dev;
		h_handle.qkey = ctx->qkey_dev;
		h_handle.signal_handles = ctx->d_signal_handles;
		h_handle.counter_handles = ctx->d_counter_handles;
		h_handle.pending_reqs = 0;
		h_handle.nranks = nranks;
		h_handle.rank = rank;

		ctx->d_handle = static_cast<nccl_ofi_gin_gdaki_dev_handle *>(
			gpu_alloc_zeroed(sizeof(nccl_ofi_gin_gdaki_dev_handle)));
		if (!ctx->d_handle) {
			throw std::runtime_error("gpu_alloc for device handle failed");
		}
		if (nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->d_handle, &h_handle, sizeof(h_handle)) != 0) {
			throw std::runtime_error("gpu_mem_copy device handle failed");
		}

		/*
		 * Step 13: Populate the host-side ncclNetDeviceHandle.
		 */
		auto *dev_handle = static_cast<ncclNetDeviceHandle_v11_t *>(
			calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
		if (dev_handle == nullptr) {
			throw std::runtime_error("calloc ncclNetDeviceHandle failed");
		}

		dev_handle->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDAKI;
		dev_handle->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
		dev_handle->handle = ctx->d_handle;
		dev_handle->size = sizeof(nccl_ofi_gin_gdaki_dev_handle);
		dev_handle->needsProxyProgress = 0;

		*ginCtx = ctx;
		*devHandle = dev_handle;

		NCCL_OFI_INFO(NCCL_NET,
			      "gin GDAKI: createContext done (nranks=%d rank=%d nSignals=%d nCounters=%d "
			      "sq_entries=%u sq_entry_size=%u cq_entries=%u cq_entry_size=%u)",
			      nranks, rank, config->nSignals, config->nCounters,
			      sq_attr.num_entries, sq_attr.entry_size,
			      efa_cq_attr.num_entries, efa_cq_attr.entry_size);

		return ncclSuccess;

	} catch (const std::exception &e) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed: %s", e.what());
		gdaki_destroy_ctx(ctx);
		return ncclSystemError;
	}
#endif /* GDAKI_ENABLED */
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
#if !GDAKI_ENABLED
	(void)ginCtx;
	return ncclSuccess;
#else
	gdaki_destroy_ctx(static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx));
	return ncclSuccess;
#endif
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
	.regMrSym = nccl_ofi_gin_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_deregMrSym,
	.destroyContext = nccl_ofi_gin_gdaki_destroyContext,
	.closeColl = nccl_ofi_gin_closeColl,
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
