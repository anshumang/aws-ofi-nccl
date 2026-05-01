/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Standalone test: GPU-initiated RDMA write via efa-dp-direct.
 *
 * Uses the plugin's regMrSym to register buffers (which does both proxy-side
 * and efa-direct registration), then launches a CUDA kernel that posts an
 * RDMA write WR directly to the EFA SQ.
 *
 * Build: see build_put_test.sh
 * Run:   OFI_NCCL_GIN_GDAKI=1 srun -N 2 --ntasks-per-node=1 --gpus-per-node=1 ./gin_put_test
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <mpi.h>
#include <dlfcn.h>
#include <cuda_runtime.h>

#include "efa_cuda_dp.h"
#include "efa_cuda_dp.cuh"

/* Minimal type defs to avoid plugin-internal headers */
typedef void (*ncclDebugLogger_t)(int level, unsigned long flags, const char *file,
				  int line, const char *fmt, ...);
typedef void (*ncclProfilerCallback_t)(void **eHandle, int type, void *pHandle, int64_t pluginId,
				       void *extData);
typedef int ncclResult_t;
#define ncclSuccess 0
#define NCCL_NET_HANDLE_MAXSIZE 256
#define NCCL_LOG_INFO 4
#define NCCL_PTR_CUDA 2

#include "nccl/net_device.h"

typedef struct {
	int nSignals; int nCounters; int nContexts; int queueDepth; int trafficClass;
} ncclGinConfig_v13_t;

typedef struct { int dummy; } ncclNetCommConfig_v11_t;

#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

#define CHECK_CUDA(call) do { \
	cudaError_t _e = (call); \
	if (_e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } \
} while(0)

/* ---- CUDA kernels ---- */

__global__ void rdma_write_kernel(
	struct efa_cuda_qp *qp,
	uint16_t dest_ah, uint16_t dest_qpn, uint32_t dest_qkey,
	uint32_t lkey, uint64_t local_addr,
	uint32_t rkey, uint64_t remote_addr, uint32_t size)
{
	printf("GPU: posting RDMA write size=%u\n", size);
	uint8_t wr_buf[64];
	memset(wr_buf, 0, sizeof(wr_buf));

	efa_cuda_init_rdma_write_wr(wr_buf, 0, rkey, remote_addr);
	efa_cuda_wr_set_remote(wr_buf, dest_ah, dest_qpn, dest_qkey);
	efa_cuda_wr_set_sge(wr_buf, lkey, local_addr, size);

	int ret = efa_cuda_start_sq_batch(qp, 1);
	if (ret != 0) { printf("GPU: start_sq_batch failed: %d\n", ret); return; }
	ret = efa_cuda_sq_batch_place_wr(qp, 0, wr_buf);
	if (ret != 0) { printf("GPU: place_wr failed: %d\n", ret); return; }
	efa_cuda_flush_sq_wrs(qp);
	printf("GPU: RDMA write posted OK\n");
}

__global__ void poll_cq_kernel(struct efa_cuda_cq *cq, int *done)
{
	*done = 0;
	for (int i = 0; i < 10000000; i++) {
		void *wc = efa_cuda_cq_poll(cq, cq->cc);
		if (wc) {
			printf("GPU: CQ completion opcode=%d err=%u\n",
			       (int)efa_cuda_wc_read_opcode(wc),
			       efa_cuda_wc_read_vendor_err(wc));
			efa_cuda_cq_pop(cq, 1);
			*done = 1;
			return;
		}
	}
	printf("GPU: CQ poll timeout\n");
}

/* ---- Host ---- */

static void test_logger(int level, unsigned long flags, const char *file,
			int line, const char *fmt, ...)
{
	if (level > NCCL_LOG_INFO) return;
	va_list ap; va_start(ap, fmt); vfprintf(stdout, fmt, ap); va_end(ap);
	fprintf(stdout, "\n"); fflush(stdout);
}

/* GIN vtable — just the fields we use */
typedef struct {
	const char *name;
	ncclResult_t (*init)(void**, uint64_t, ncclDebugLogger_t);
	ncclResult_t (*devices)(int*);
	ncclResult_t (*getProperties)(int, void*);
	ncclResult_t (*listen)(void*, int, void*, void**);
	ncclResult_t (*connect)(void*, void**, int, int, void*, void**);
	ncclResult_t (*createContext)(void*, ncclGinConfig_v13_t*, void**, ncclNetDeviceHandle_v11_t**);
	ncclResult_t (*regMrSym)(void*, void*, size_t, int, uint64_t, void**, void**);
	ncclResult_t (*regMrSymDmaBuf)(void*, void*, size_t, int, uint64_t, int, uint64_t, void**, void**);
	ncclResult_t (*deregMrSym)(void*, void*);
	ncclResult_t (*destroyContext)(void*);
	ncclResult_t (*closeColl)(void*);
	ncclResult_t (*closeListen)(void*);
} gin_vtable_t;

int main(int argc, char **argv)
{
	int rank, nranks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	if (nranks < 2) { fprintf(stderr, "Need >=2 ranks\n"); MPI_Finalize(); return 1; }
	CHECK_CUDA(cudaSetDevice(0));

	/* Load plugin */
	void *dl = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	assert(dl);
	auto *extGin = (gin_vtable_t *)dlsym(dl, "ncclGinPlugin_v13");
	assert(extGin);

	/* Also need net init */
	typedef struct { const char *name; ncclResult_t (*init)(void**, uint64_t, ncclNetCommConfig_v11_t*, ncclDebugLogger_t, ncclProfilerCallback_t); } net_init_t;
	auto *extNet = (net_init_t *)dlsym(dl, "ncclNetPlugin_v11");
	assert(extNet);

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	assert(extNet->init(&netCtx, 0, &netConfig, (ncclDebugLogger_t)test_logger, nullptr) == 0);

	void *ginCtx = nullptr;
	assert(extGin->init(&ginCtx, 0, (ncclDebugLogger_t)test_logger) == 0);
	printf("Rank %d: plugin=%s\n", rank, extGin->name);

	/* Listen + connect */
	int ndev; extGin->devices(&ndev);
	char handle_buf[NCCL_NET_HANDLE_MAXSIZE];
	void *listenComm = nullptr;
	extGin->listen(ginCtx, 0, handle_buf, &listenComm);

	std::vector<char> all_handles(nranks * NCCL_NET_HANDLE_MAXSIZE);
	memcpy(&all_handles[rank * NCCL_NET_HANDLE_MAXSIZE], handle_buf, NCCL_NET_HANDLE_MAXSIZE);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_handles.data(),
		      NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, MPI_COMM_WORLD);
	std::vector<void *> hptrs(nranks);
	for (int i = 0; i < nranks; i++) hptrs[i] = &all_handles[i * NCCL_NET_HANDLE_MAXSIZE];

	void *collComm = nullptr;
	assert(extGin->connect(ginCtx, hptrs.data(), nranks, rank, listenComm, &collComm) == 0);

	/* createContext */
	ncclGinConfig_v13_t ginConfig = {0, 0, 1, 64, -1};
	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	assert(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle) == 0);
	printf("Rank %d: createContext done\n", rank);

	/* Read device handle */
	nccl_ofi_gin_gdaki_dev_handle h_dev = {};
	CHECK_CUDA(cudaMemcpy(&h_dev, devHandle->handle, sizeof(h_dev), cudaMemcpyDeviceToHost));

	/* Register test buffers via regMrSym */
	const uint32_t SZ = 64;
	const uint8_t PAT = 0xAB;
	uint8_t *src_d, *dst_d;
	CHECK_CUDA(cudaMalloc(&src_d, SZ));
	CHECK_CUDA(cudaMalloc(&dst_d, SZ));
	if (rank == 0) CHECK_CUDA(cudaMemset(src_d, PAT, SZ));
	CHECK_CUDA(cudaMemset(dst_d, 0, SZ));

	void *src_mhandle = nullptr, *src_ginhandle = nullptr;
	void *dst_mhandle = nullptr, *dst_ginhandle = nullptr;
	assert(extGin->regMrSym(collComm, src_d, SZ, NCCL_PTR_CUDA, 0, &src_mhandle, &src_ginhandle) == 0);
	assert(extGin->regMrSym(collComm, dst_d, SZ, NCCL_PTR_CUDA, 0, &dst_mhandle, &dst_ginhandle) == 0);
	printf("Rank %d: regMrSym done src_ginhandle=%p dst_ginhandle=%p\n",
	       rank, src_ginhandle, dst_ginhandle);

	/* Extract lkey and rkeys from the GDAKI MR handle.
	 * ginHandle is a nccl_ofi_gin_gdaki_mr_reg* whose ->handle has lkey + rkeys[]. */
	struct quick_mr_reg { void *mr; nccl_ofi_gin_gdaki_mr_handle *handle; };
	auto *src_reg = (quick_mr_reg *)src_ginhandle;
	auto *dst_reg = (quick_mr_reg *)dst_ginhandle;

	uint32_t src_lkey = src_reg->handle->lkey;
	printf("Rank %d: src_lkey=0x%x dst_rkeys[0]=0x%x\n",
	       rank, src_lkey, dst_reg->handle->rkeys[0]);

	/* Read per-peer addressing from GPU */
	std::vector<uint16_t> h_ahs(nranks), h_qpns(nranks);
	std::vector<uint32_t> h_qkeys(nranks);
	CHECK_CUDA(cudaMemcpy(h_ahs.data(), h_dev.address_handles, nranks * 2, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_qpns.data(), h_dev.remote_qpns, nranks * 2, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_qkeys.data(), h_dev.qkey, nranks * 4, cudaMemcpyDeviceToHost));

	/* Allgather dst GPU VAs (not symmetric heap — each rank has different VA) */
	std::vector<uint64_t> all_vas(nranks);
	all_vas[rank] = (uint64_t)dst_d;
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_vas.data(),
		      sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 0: RDMA write to rank 1 */
	if (rank == 0) {
		int tgt = 1;
		uint32_t rkey = dst_reg->handle->rkeys[tgt];
		printf("R0: write to R%d ah=%u qpn=%u qkey=%u lkey=0x%x rkey=0x%x va=0x%lx\n",
		       tgt, h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt], src_lkey, rkey, all_vas[tgt]);

		rdma_write_kernel<<<1,1>>>(h_dev.qp, h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt],
					   src_lkey, (uint64_t)src_d, rkey, all_vas[tgt], SZ);
		CHECK_CUDA(cudaDeviceSynchronize());

		int *d_done; CHECK_CUDA(cudaMalloc(&d_done, 4));
		poll_cq_kernel<<<1,1>>>(h_dev.cq, d_done);
		CHECK_CUDA(cudaDeviceSynchronize());
		int h_done = 0;
		CHECK_CUDA(cudaMemcpy(&h_done, d_done, 4, cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaFree(d_done));
		printf("R0: CQ poll %s\n", h_done ? "OK" : "TIMEOUT");
	}

	MPI_Barrier(MPI_COMM_WORLD);
	sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 1: verify */
	if (rank == 1) {
		uint8_t buf[SZ];
		CHECK_CUDA(cudaMemcpy(buf, dst_d, SZ, cudaMemcpyDeviceToHost));
		bool ok = true;
		for (uint32_t i = 0; i < SZ; i++) {
			if (buf[i] != PAT) {
				fprintf(stderr, "R1: FAIL byte %u: 0x%02x != 0x%02x\n", i, buf[i], PAT);
				ok = false; break;
			}
		}
		printf("R1: %s\n", ok ? "PASS" : "FAIL");
	}

	/* Cleanup */
	extGin->deregMrSym(collComm, dst_mhandle);
	extGin->deregMrSym(collComm, src_mhandle);
	CHECK_CUDA(cudaFree(dst_d));
	CHECK_CUDA(cudaFree(src_d));
	extGin->destroyContext(proxyCtx);
	extGin->closeColl(collComm);
	extGin->closeListen(listenComm);

	dlclose(dl);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	printf("Rank %d: done\n", rank);
	return 0;
}
