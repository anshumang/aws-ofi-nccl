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
#include <cuda_runtime.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_ctx.h"
#include "nccl_ofi_ofiutils.h"

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
	props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
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
 * Helper: CUDA-allocate and zero a buffer. Returns nullptr on failure.
 */
static void *cuda_alloc_zeroed(size_t size)
{
	void *ptr = nullptr;
	cudaError_t err = cudaMalloc(&ptr, size);
	if (err != cudaSuccess) {
		return nullptr;
	}
	err = cudaMemset(ptr, 0, size);
	if (err != cudaSuccess) {
		cudaFree(ptr);
		return nullptr;
	}
	return ptr;
}

/*
 * Helper: free a CUDA pointer if non-null.
 */
static void cuda_free(void *ptr)
{
	if (ptr) {
		cudaFree(ptr);
	}
}

/*
 * Tear down everything in ctx. Safe to call on a partially-initialized ctx.
 */
static void gdaki_destroy_ctx(nccl_ofi_gin_gdaki_context *ctx)
{
	if (!ctx) {
		return;
	}

	/* 1. Free per-peer GPU arrays */
	cuda_free(ctx->address_handles_dev);
	cuda_free(ctx->remote_qpns_dev);
	cuda_free(ctx->qkey_dev);

	/* 3. Free device handle struct */
	cuda_free(ctx->d_handle);

	/* 4. Deregister and free SQ/CQ GPU buffers */
	if (ctx->sq_mr) {
		fi_close(&ctx->sq_mr->fid);
	}
	cuda_free(ctx->sq_buffer_dev);
	if (ctx->cq_mr) {
		fi_close(&ctx->cq_mr->fid);
	}
	cuda_free(ctx->cq_buffer_dev);

	/* 5. Close libfabric resources (order matters: ep before cq/av/domain/fabric) */
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
		 * Step 7: Store raw SQ/RQ/CQ attributes in the device handle
		 * for NCCL's host code to create efa-dp-direct device objects.
		 */
		nccl_ofi_gin_gdaki_wq_attrs h_sq_attrs = {};
		h_sq_attrs.buffer = sq_attr.buffer;
		h_sq_attrs.entry_size = sq_attr.entry_size;
		h_sq_attrs.num_entries = sq_attr.num_entries;
		h_sq_attrs.doorbell = sq_attr.doorbell;
		h_sq_attrs.max_batch = sq_attr.max_batch;

		nccl_ofi_gin_gdaki_wq_attrs h_rq_attrs = {};
		h_rq_attrs.buffer = rq_attr.buffer;
		h_rq_attrs.entry_size = rq_attr.entry_size;
		h_rq_attrs.num_entries = rq_attr.num_entries;
		h_rq_attrs.doorbell = rq_attr.doorbell;
		h_rq_attrs.max_batch = rq_attr.max_batch;

		nccl_ofi_gin_gdaki_cq_attrs h_cq_attrs = {};
		h_cq_attrs.buffer = efa_cq_attr.buffer;
		h_cq_attrs.entry_size = efa_cq_attr.entry_size;
		h_cq_attrs.num_entries = efa_cq_attr.num_entries;

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

		ctx->address_handles_dev = static_cast<uint16_t *>(cuda_alloc_zeroed(ahn_size));
		ctx->remote_qpns_dev = static_cast<uint16_t *>(cuda_alloc_zeroed(qpn_size));
		ctx->qkey_dev = static_cast<uint32_t *>(cuda_alloc_zeroed(qkey_size));
		if (!ctx->address_handles_dev || !ctx->remote_qpns_dev || !ctx->qkey_dev) {
			throw std::runtime_error("cudaMalloc for per-peer arrays failed");
		}

		cudaError_t cu_ret;
		cu_ret = cudaMemcpy(ctx->address_handles_dev, h_ahns.data(), ahn_size, cudaMemcpyHostToDevice);
		if (cu_ret != cudaSuccess) throw std::runtime_error("cudaMemcpy ahns failed");
		cu_ret = cudaMemcpy(ctx->remote_qpns_dev, h_qpns.data(), qpn_size, cudaMemcpyHostToDevice);
		if (cu_ret != cudaSuccess) throw std::runtime_error("cudaMemcpy qpns failed");
		cu_ret = cudaMemcpy(ctx->qkey_dev, h_qkeys.data(), qkey_size, cudaMemcpyHostToDevice);
		if (cu_ret != cudaSuccess) throw std::runtime_error("cudaMemcpy qkeys failed");

		/*
		 * Step 11: Populate and upload the device handle.
		 */
		nccl_ofi_gin_gdaki_dev_handle h_handle = {};
		h_handle.sq_attrs = h_sq_attrs;
		h_handle.rq_attrs = h_rq_attrs;
		h_handle.cq_attrs = h_cq_attrs;
		h_handle.address_handles = ctx->address_handles_dev;
		h_handle.remote_qpns = ctx->remote_qpns_dev;
		h_handle.qkey = ctx->qkey_dev;
		h_handle.signal_handles = nullptr;
		h_handle.counter_handles = nullptr;
		h_handle.pending_reqs = 0;
		h_handle.nranks = nranks;
		h_handle.rank = rank;

		ctx->d_handle = static_cast<nccl_ofi_gin_gdaki_dev_handle *>(
			cuda_alloc_zeroed(sizeof(nccl_ofi_gin_gdaki_dev_handle)));
		if (!ctx->d_handle) {
			throw std::runtime_error("cudaMalloc for device handle failed");
		}
		cu_ret = cudaMemcpy(ctx->d_handle, &h_handle, sizeof(h_handle), cudaMemcpyHostToDevice);
		if (cu_ret != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy device handle failed");
		}

		/*
		 * Step 12: Populate the host-side ncclNetDeviceHandle.
		 */
		auto *dev_handle = static_cast<ncclNetDeviceHandle_v11_t *>(
			calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
		if (dev_handle == nullptr) {
			throw std::runtime_error("calloc ncclNetDeviceHandle failed");
		}

		dev_handle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
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
