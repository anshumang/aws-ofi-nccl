/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin for the GIN API. Shared APIs (init, devices, listen, connect,
 * regMrSym[DmaBuf], deregMrSym, closeColl, closeListen, ginProgress, finalize)
 * are reused from the proxy-side implementations in nccl_ofi_gin_api.cpp.
 * Only the GDAKI-specific stubs (createContext/destroyContext/get_properties/
 * regMr*-return-error/queryLastError) live here until they are implemented.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

#if HAVE_EFA_DP_DIRECT
#include <cuda_runtime.h>

#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_ctx.h"
#endif /* HAVE_EFA_DP_DIRECT */

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

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
#if !HAVE_EFA_DP_DIRECT
	(void)collComm;
	(void)config;
	(void)ginCtx;
	(void)devHandle;
	NCCL_OFI_WARN("gin GDAKI: createContext requires efa-dp-direct; plugin was built without --with-efa-dp-direct");
	return ncclInternalError;
#else
	if (collComm == nullptr || config == nullptr || ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);

	/*
	 * Allocate the host-side context. This struct owns every libfabric
	 * fid, every fi_mr_reg registration, and every CUDA allocation that
	 * createContext is about to make.
	 */
	auto *ctx = new (std::nothrow) nccl_ofi_gin_gdaki_context();
	if (ctx == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed to allocate context");
		return ncclSystemError;
	}

	ctx->nSignals = config->nSignals;
	ctx->nCounters = config->nCounters;
	ctx->nranks = put_comm->get_nranks();
	ctx->rank = put_comm->get_rank();

	/*
	 * Stub: allocate the device-visible handle in GPU memory
	 * and leave the libfabric/efa-dp-direct/per-peer-array members zeroed.
	 * Subsequent patches fill in ofi_fabric/domain/ep/av/cq, the GDA ops
	 * extension, the efa_cuda_qp/cq objects, and the per-peer addressing
	 * arrays.
	 *
	 * Publishing the device handle now gives the kernel-side implementation
	 * Put/PutValue implementation a fixed layout to code against before the
	 * data-plane resources are populated.
	 */
	nccl_ofi_gin_gdaki_dev_handle h_handle = {};
	h_handle.nranks = ctx->nranks;
	h_handle.rank = ctx->rank;

	cudaError_t cu_ret = cudaMalloc(reinterpret_cast<void **>(&ctx->d_handle),
					sizeof(nccl_ofi_gin_gdaki_dev_handle));
	if (cu_ret != cudaSuccess) {
		NCCL_OFI_WARN("gin GDAKI: cudaMalloc for device handle failed: %s",
			      cudaGetErrorString(cu_ret));
		delete ctx;
		return ncclSystemError;
	}

	cu_ret = cudaMemcpy(ctx->d_handle, &h_handle, sizeof(h_handle), cudaMemcpyHostToDevice);
	if (cu_ret != cudaSuccess) {
		NCCL_OFI_WARN("gin GDAKI: cudaMemcpy of device handle failed: %s",
			      cudaGetErrorString(cu_ret));
		cudaFree(ctx->d_handle);
		delete ctx;
		return ncclSystemError;
	}

	/*
	 * Allocate and populate the host-side ncclNetDeviceHandle. NCCL frees
	 * this pointer (or rather, its owning wrapper) after destroyContext
	 * returns.
	 */
	auto *dev_handle = static_cast<ncclNetDeviceHandle_v11_t *>(
		calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
	if (dev_handle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: failed to allocate ncclNetDeviceHandle");
		cudaFree(ctx->d_handle);
		delete ctx;
		return ncclSystemError;
	}

	dev_handle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
	dev_handle->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	dev_handle->handle = ctx->d_handle;
	dev_handle->size = sizeof(nccl_ofi_gin_gdaki_dev_handle);
	dev_handle->needsProxyProgress = 0;

	*ginCtx = ctx;
	*devHandle = dev_handle;

	NCCL_OFI_INFO(NCCL_NET,
		      "gin GDAKI: createContext produced stub device handle (nranks=%d rank=%d nSignals=%d nCounters=%d)",
		      ctx->nranks, ctx->rank, ctx->nSignals, ctx->nCounters);

	return ncclSuccess;
#endif /* HAVE_EFA_DP_DIRECT */
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
#if !HAVE_EFA_DP_DIRECT
	(void)ginCtx;
	return ncclSuccess;
#else
	if (ginCtx == nullptr) {
		return ncclSuccess;
	}

	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx);

	/*
	 * Tear-down is ordered to mirror createContext:
	 *   1. Destroy efa-dp-direct device objects (qp, cq) — populated by
	 *      later patches; no-op in the stub.
	 *   2. Free per-peer GPU arrays — no-op in the stub.
	 *   3. Free GPU memory backing the device handle struct.
	 *   4. Release libfabric resources (ep, cq, av, domain, fabric) — no-op
	 *      in the stub.
	 *   5. Free the host-side context.
	 */

	if (ctx->d_handle != nullptr) {
		cudaError_t cu_ret = cudaFree(ctx->d_handle);
		if (cu_ret != cudaSuccess) {
			NCCL_OFI_WARN("gin GDAKI: cudaFree of device handle failed: %s",
				      cudaGetErrorString(cu_ret));
		}
		ctx->d_handle = nullptr;
	}

	delete ctx;

	return ncclSuccess;
#endif /* HAVE_EFA_DP_DIRECT */
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
