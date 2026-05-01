/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Standalone test: GPU-initiated RDMA write via efa-dp-direct.
 *
 * Rank 0 writes a known pattern to rank 1's buffer using a CUDA kernel
 * that posts an RDMA write WR directly to the EFA SQ. Rank 1 verifies.
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

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include "efa_cuda_dp.h"
#include "efa_cuda_dp.cuh"

/* Minimal type defs — avoid pulling in plugin internals */
typedef void (*ncclDebugLogger_t)(int level, unsigned long flags, const char *file,
				  int line, const char *fmt, ...);
typedef void (*ncclProfilerCallback_t)(void **eHandle, int type, void *pHandle, int64_t pluginId,
				       void *extData);
typedef int ncclResult_t;
#define ncclSuccess 0
#define NCCL_NET_HANDLE_MAXSIZE 256
#define NCCL_LOG_INFO 4

#include "nccl/net_device.h"

/* GIN v13 vtable — just the fields we use */
typedef struct {
	int nSignals; int nCounters; int nContexts; int queueDepth; int trafficClass;
} ncclGinConfig_v13_t;

typedef struct { int dummy; } ncclNetCommConfig_v11_t;

/* Our device handle */
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
	printf("GPU: kernel entered qp=%p\n", qp);
	if (!qp) { printf("GPU: qp is NULL!\n"); return; }

	/* Verify QP is accessible */
	printf("GPU: qp->sq.wq.max_wqes=%u sq.wq.pc=%u sq.wq.max_batch=%u\n",
	       qp->sq.wq.max_wqes, qp->sq.wq.pc, qp->sq.wq.max_batch);
	printf("GPU: sq.wq.buf=%p sq.wq.db=%p rq.wq.buf=%p\n",
	       qp->sq.wq.buf, qp->sq.wq.db, qp->rq.wq.buf);

	bool compat = efa_cuda_is_qp_compatible(qp);
	printf("GPU: qp compatible=%d\n", (int)compat);
	if (!compat) { printf("GPU: QP not compatible!\n"); return; }

	uint8_t wr_buf[64];
	memset(wr_buf, 0, sizeof(wr_buf));

	efa_cuda_init_rdma_write_wr(wr_buf, 0, rkey, remote_addr);
	efa_cuda_wr_set_remote(wr_buf, dest_ah, dest_qpn, dest_qkey);
	efa_cuda_wr_set_sge(wr_buf, lkey, local_addr, size);
	printf("GPU: WR prepared\n");

	int ret = efa_cuda_start_sq_batch(qp, 1);
	printf("GPU: start_sq_batch ret=%d\n", ret);
	if (ret != 0) return;

	ret = efa_cuda_sq_batch_place_wr(qp, 0, wr_buf);
	printf("GPU: place_wr ret=%d\n", ret);
	if (ret != 0) return;

	efa_cuda_flush_sq_wrs(qp);
	printf("GPU: RDMA write posted (size=%u)\n", size);
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

int main(int argc, char **argv)
{
	int rank, nranks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	if (nranks < 2) { fprintf(stderr, "Need >=2 ranks\n"); MPI_Finalize(); return 1; }
	CHECK_CUDA(cudaSetDevice(0));

	/* Load plugin via dlopen */
	void *dl = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	assert(dl);

	/* We need: net init/finalize, gin init/devices/listen/connect/createContext/destroyContext/closeColl/closeListen/finalize */
	typedef struct {
		const char *name;
		ncclResult_t (*init)(void**, uint64_t, ncclNetCommConfig_v11_t*, ncclDebugLogger_t, ncclProfilerCallback_t);
		/* ... we only need init and finalize */
		ncclResult_t (*pad[30])(void);  /* skip to finalize — fragile but works for this test */
		ncclResult_t (*finalize)(void*);
	} net_vtable_t;

	/* Actually, let's just grab the gin vtable which has everything we need */
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
		/* remaining fields not needed */
	} gin_vtable_t;

	/* Net vtable — we only need init and finalize */
	typedef struct {
		const char *name;
		ncclResult_t (*init)(void**, uint64_t, ncclNetCommConfig_v11_t*, ncclDebugLogger_t, ncclProfilerCallback_t);
	} net_init_t;

	auto *extGin = (gin_vtable_t *)dlsym(dl, "ncclGinPlugin_v13");
	auto *extNet_init = (net_init_t *)dlsym(dl, "ncclNetPlugin_v11");
	assert(extGin && extNet_init);

	/* Init net (required before gin) */
	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	assert(extNet_init->init(&netCtx, 0, &netConfig, (ncclDebugLogger_t)test_logger, nullptr) == 0);

	/* Init gin */
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
	assert(devHandle && devHandle->handle);
	printf("Rank %d: createContext done\n", rank);

	/* Read device handle back to host */
	nccl_ofi_gin_gdaki_dev_handle h_dev = {};
	CHECK_CUDA(cudaMemcpy(&h_dev, devHandle->handle, sizeof(h_dev), cudaMemcpyDeviceToHost));

	/* Read QP struct back to check buffer pointers */
	efa_cuda_qp h_qp = {};
	CHECK_CUDA(cudaMemcpy(&h_qp, h_dev.qp, sizeof(h_qp), cudaMemcpyDeviceToHost));
	printf("Rank %d: QP sq.wq.buf=%p sq.wq.db=%p sq.wq.max_wqes=%u sq.wq.max_batch=%u\n",
	       rank, h_qp.sq.wq.buf, h_qp.sq.wq.db, h_qp.sq.wq.max_wqes, h_qp.sq.wq.max_batch);

	/* Check if SQ buffer is GPU-accessible */
	cudaPointerAttributes sq_attrs;
	cudaError_t pa_ret = cudaPointerGetAttributes(&sq_attrs, h_qp.sq.wq.buf);
	printf("Rank %d: SQ buf pointer attrs: type=%d device=%d (err=%s)\n",
	       rank, (int)sq_attrs.type, sq_attrs.device,
	       pa_ret == cudaSuccess ? "ok" : cudaGetErrorString(pa_ret));

	/* Try to register the SQ/CQ MMIO regions with CUDA for GPU access */
	{
		/* SQ buffer: entry_size * num_entries */
		size_t sq_size = (size_t)h_qp.sq.wq.max_wqes * 64; /* entry_size from query was 64 */
		cudaError_t reg_ret = cudaHostRegister(h_qp.sq.wq.buf, sq_size, cudaHostRegisterIoMemory);
		printf("Rank %d: cudaHostRegister SQ buf (%zu bytes): %s\n",
		       rank, sq_size, reg_ret == cudaSuccess ? "ok" : cudaGetErrorString(reg_ret));

		/* Doorbell */
		reg_ret = cudaHostRegister(h_qp.sq.wq.db, 4096, cudaHostRegisterIoMemory);
		printf("Rank %d: cudaHostRegister SQ doorbell: %s\n",
		       rank, reg_ret == cudaSuccess ? "ok" : cudaGetErrorString(reg_ret));

		/* CQ buffer */
		efa_cuda_cq h_cq = {};
		CHECK_CUDA(cudaMemcpy(&h_cq, h_dev.cq, sizeof(h_cq), cudaMemcpyDeviceToHost));
		size_t cq_size = (size_t)h_cq.num_entries * h_cq.entry_size;
		reg_ret = cudaHostRegister(h_cq.buf, cq_size, cudaHostRegisterIoMemory);
		printf("Rank %d: cudaHostRegister CQ buf (%zu bytes): %s\n",
		       rank, cq_size, reg_ret == cudaSuccess ? "ok" : cudaGetErrorString(reg_ret));
	}

	/* Access the efa-direct domain from the context for MR registration.
	 * The context struct layout: fabric, domain, ep, av, cq, info, ... */
	struct { fid_fabric *f; fid_domain *d; fid_ep *e; fid_av *a; fid_cq *c; fi_info *i; } *qctx;
	qctx = decltype(qctx)(proxyCtx);

	/* Get GDA ops for get_mr_lkey */
	fi_efa_ops_gda *gda_ops = nullptr;
	assert(fi_open_ops(&qctx->d->fid, FI_EFA_GDA_OPS, 0, (void **)&gda_ops, nullptr) == 0);

	/* Allocate and register test buffers */
	const uint32_t SZ = 64;
	const uint8_t PAT = 0xAB;
	uint8_t *src_d, *dst_d;
	CHECK_CUDA(cudaMalloc(&src_d, SZ));
	CHECK_CUDA(cudaMalloc(&dst_d, SZ));
	if (rank == 0) CHECK_CUDA(cudaMemset(src_d, PAT, SZ));
	CHECK_CUDA(cudaMemset(dst_d, 0, SZ));

	auto reg_mr = [&](void *buf, fid_mr **mr) {
		iovec iov = {buf, SZ};
		fi_mr_attr attr = {};
		attr.mr_iov = &iov;
		attr.iov_count = 1;
		attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
		int ret = fi_mr_regattr(qctx->d, &attr, 0, mr);
		if (ret) { fprintf(stderr, "R%d: fi_mr_regattr failed: %s\n", rank, fi_strerror(-ret)); exit(1); }
	};

	fid_mr *src_mr, *dst_mr;
	reg_mr(src_d, &src_mr);
	reg_mr(dst_d, &dst_mr);

	uint64_t src_lkey = gda_ops->get_mr_lkey(src_mr);
	uint64_t dst_rkey = fi_mr_key(dst_mr);
	printf("Rank %d: src_lkey=0x%lx dst_rkey=0x%lx\n", rank, src_lkey, dst_rkey);

	/* Allgather rkeys and dst GPU VAs */
	std::vector<uint64_t> all_rkeys(nranks), all_vas(nranks);
	all_rkeys[rank] = dst_rkey;
	all_vas[rank] = (uint64_t)dst_d;
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_rkeys.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_vas.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);

	/* Read per-peer addressing from GPU */
	std::vector<uint16_t> h_ahs(nranks), h_qpns(nranks);
	std::vector<uint32_t> h_qkeys(nranks);
	CHECK_CUDA(cudaMemcpy(h_ahs.data(), h_dev.address_handles, nranks * 2, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_qpns.data(), h_dev.remote_qpns, nranks * 2, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_qkeys.data(), h_dev.qkey, nranks * 4, cudaMemcpyDeviceToHost));

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 0: RDMA write to rank 1 */
	if (rank == 0) {
		int tgt = 1;
		printf("R0: write to R%d ah=%u qpn=%u qkey=%u rkey=0x%lx va=0x%lx\n",
		       tgt, h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt], all_rkeys[tgt], all_vas[tgt]);

		rdma_write_kernel<<<1,1>>>(h_dev.qp, h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt],
					   (uint32_t)src_lkey, (uint64_t)src_d,
					   (uint32_t)all_rkeys[tgt], all_vas[tgt], SZ);
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
	fi_close(&dst_mr->fid);
	fi_close(&src_mr->fid);
	CHECK_CUDA(cudaFree(dst_d));
	CHECK_CUDA(cudaFree(src_d));
	extGin->destroyContext(proxyCtx);
	extGin->closeColl(collComm);
	extGin->closeListen(listenComm);

	/* finalize gin — need the actual function pointer. It's at a known offset in the vtable.
	 * gin_v13 has 22 function pointers after name. finalize is the last one. */
	typedef ncclResult_t (*finalize_fn)(void*);
	auto *raw = (void **)extGin;
	/* name=0, init=1, devices=2, getProperties=3, listen=4, connect=5, createContext=6,
	   regMrSym=7, regMrSymDmaBuf=8, deregMrSym=9, destroyContext=10, closeColl=11,
	   closeListen=12, iput=13, iputSignal=14, iget=15, iflush=16, test=17,
	   ginProgress=18, queryLastError=19, finalize=20 */
	auto gin_finalize = (finalize_fn)raw[21]; /* +1 for name pointer */
	gin_finalize(ginCtx);

	/* net finalize — similarly at a known offset. Just dlsym the symbol and offset. */
	/* Skip net finalize for now — process is exiting anyway */

	dlclose(dl);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	printf("Rank %d: done\n", rank);
	return 0;
}
