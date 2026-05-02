/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * CPU-side RDMA write test using efa-direct GDA ops.
 *
 * Rank 0 posts an RDMA write to rank 1's buffer by writing directly to the
 * EFA SQ MMIO region (no GPU, no efa-dp-direct). Uses only libfabric's
 * efa-direct provider and the FI_EFA_GDA_OPS extension.
 *
 * Run with at least 2 MPI ranks.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <string.h>
#include <vector>

#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>

#ifdef HAVE_RDMA_FI_EXT_EFA_H
#include <rdma/fi_ext_efa.h>
#endif

#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_ctx.h"

/* EFA hardware I/O definitions for direct SQ/CQ access. */

enum efa_io_queue_type {
	/* send queue (of a QP) */
	EFA_IO_SEND_QUEUE                           = 1,
	/* recv queue (of a QP) */
	EFA_IO_RECV_QUEUE                           = 2,
};

enum efa_io_send_op_type {
	/* send message */
	EFA_IO_SEND                                 = 0,
	/* RDMA read */
	EFA_IO_RDMA_READ                            = 1,
	/* RDMA write */
	EFA_IO_RDMA_WRITE                           = 2,
};

enum efa_io_comp_status {
	/* Successful completion */
	EFA_IO_COMP_STATUS_OK                       = 0,
	/* Flushed during QP destroy */
	EFA_IO_COMP_STATUS_FLUSHED                  = 1,
	/* Internal QP error */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR = 2,
	/* Unsupported operation */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP = 3,
	/* Bad AH */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH   = 4,
	/* LKEY not registered or does not match IOVA */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY = 5,
	/* Message too long */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH   = 6,
	/* RKEY not registered or does not match remote IOVA */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS = 7,
	/* Connection was reset by remote side */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT       = 8,
	/* Bad dest QP number (QP does not exist or is in error state) */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN = 9,
	/* Destination resource not ready (no WQEs posted on RQ) */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR         = 10,
	/* Receiver SGL too short */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH  = 11,
	/* Unexpected status returned by responder */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS  = 12,
	/* Unresponsive remote - was previously responsive */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE = 13,
	/* No valid AH at remote side (required for RDMA operations) */
	EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER = 14,
	/* Unreachable remote - never received a response */
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE = 15,
};

/* Bitfield helpers (C++ compatible) */
#define BIT(nr) (1UL << (nr))
#define GENMASK(h, l) (((~0UL) - (1UL << (l)) + 1) & (~0UL >> (63 - (h))))
#define __bf_shf(x) (__builtin_ffsll(x) - 1)
#define FIELD_PREP(mask, val) ((((unsigned long)(val)) << __bf_shf(mask)) & (mask))
#define FIELD_GET(mask, reg) ((((unsigned long)(reg)) & (mask)) >> __bf_shf(mask))
#define EFA_GET(ptr, mask) FIELD_GET(mask##_MASK, *(ptr))
#define EFA_SET(ptr, mask, value) \
	do { auto *_p = (ptr); *_p = (*_p & ~(mask##_MASK)) | FIELD_PREP(mask##_MASK, value); } while(0)

#define EFA_IO_TX_META_DESC_OP_TYPE_MASK    GENMASK(3, 0)
#define EFA_IO_TX_META_DESC_META_DESC_MASK  BIT(7)
#define EFA_IO_TX_META_DESC_PHASE_MASK      BIT(0)
#define EFA_IO_TX_META_DESC_FIRST_MASK      BIT(2)
#define EFA_IO_TX_META_DESC_LAST_MASK       BIT(3)
#define EFA_IO_TX_META_DESC_COMP_REQ_MASK   BIT(4)
#define EFA_IO_TX_BUF_DESC_LKEY_MASK        GENMASK(23, 0)
#define EFA_IO_CDESC_COMMON_PHASE_MASK      BIT(0)
#define EFA_IO_CDESC_COMMON_Q_TYPE_MASK     GENMASK(2, 1)
#define EFA_IO_CDESC_COMMON_OP_TYPE_MASK    GENMASK(6, 4)

struct efa_io_tx_meta_desc {
	uint16_t req_id;
	uint8_t ctrl1;
	uint8_t ctrl2;
	uint16_t dest_qp_num;
	uint16_t length;
	uint32_t immediate_data;
	uint16_t ah;
	uint16_t reserved;
	uint32_t qkey;
	uint8_t reserved2[12];
};

struct efa_io_tx_buf_desc {
	uint32_t length;
	uint32_t lkey;
	uint32_t buf_addr_lo;
	uint32_t buf_addr_hi;
};

struct efa_io_remote_mem_addr {
	uint32_t length;
	uint32_t rkey;
	uint32_t buf_addr_lo;
	uint32_t buf_addr_hi;
};

struct efa_io_rdma_req {
	struct efa_io_remote_mem_addr remote_mem;
	struct efa_io_tx_buf_desc local_mem[1];
};

struct efa_io_tx_wqe {
	struct efa_io_tx_meta_desc meta;
	union {
		struct efa_io_tx_buf_desc sgl[2];
		uint8_t inline_data[32];
		struct efa_io_rdma_req rdma_req;
	} data;
};

struct efa_io_cdesc_common {
	uint16_t req_id;
	uint8_t status;
	uint8_t flags;
	uint16_t qp_num;
};

/* MMIO helpers */
#define mmio_flush_writes() asm volatile("sfence" ::: "memory")
#define udma_from_device_barrier() asm volatile("lfence" ::: "memory")

static inline void mmio_write32(void *addr, uint32_t value)
{
	__atomic_store_n((volatile uint32_t *)addr, value, __ATOMIC_RELAXED);
}

/* Copy 64 bytes to MMIO in order */
static inline void mmio_memcpy_x64(void *dest, const void *src, size_t bytecnt)
{
	const uint64_t *s = (const uint64_t *)src;
	volatile uint64_t *d = (volatile uint64_t *)dest;
	for (size_t i = 0; i < bytecnt / 8; i++) {
		__atomic_store_n(&d[i], s[i], __ATOMIC_RELAXED);
	}
}

/* ---- Test helpers ---- */

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char *argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, nranks, proc_name_len, local_rank = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Need at least 2 ranks");
		MPI_Finalize();
		return 1;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)], &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) ++local_rank;
		}
	}

	CUDACHECK(cudaSetDevice(local_rank));

	/* Load plugin */
	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (!extNet || !extGin) { MPI_Finalize(); return 1; }

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));

	int ndev;
	OFINCCLCHECK(extGin->devices(&ndev));
	int dev = local_rank % ndev;

	/* Listen + connect */
	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(),
		      NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) handles_ptrs[i] = &handles[i];

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank, listenComm, &collComm));

	/* createContext */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: createContext done", rank);

	/*
	 * Access the efa-direct domain from the context to open GDA ops
	 * and register MRs. The context struct layout starts with:
	 *   fid_fabric*, fid_domain*, fid_ep*, fid_av*, fid_cq*, fi_info*
	 */

#if HAVE_DECL_FI_EFA_GDA_OPS
	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(proxyCtx);

	/* Read device handle to get per-peer addressing */
	nccl_ofi_gin_gdaki_dev_handle h_dev = {};
	CUDACHECK(cudaMemcpy(&h_dev, devHandle->handle, sizeof(h_dev), cudaMemcpyDeviceToHost));

	/* Open GDA ops for MR registration and direct SQ/CQ access */
	struct fi_efa_ops_gda *gda_ops = nullptr;
	int ret = fi_open_ops(&ctx->ofi_domain->fid, FI_EFA_GDA_OPS, 0, (void **)&gda_ops, nullptr);
	if (ret != 0 || !gda_ops) {
		NCCL_OFI_WARN("fi_open_ops FI_EFA_GDA_OPS failed: %s", fi_strerror(-ret));
		MPI_Finalize();
		return 1;
	}

	/* Query SQ/CQ attrs for CPU-side direct access (host pointers) */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	ret = gda_ops->query_qp_wqs(ctx->ofi_ep, &sq_attr, &rq_attr);
	if (ret != 0) { NCCL_OFI_WARN("query_qp_wqs failed"); MPI_Finalize(); return 1; }

	struct fi_efa_cq_attr efa_cq_attr = {};
	ret = gda_ops->query_cq(ctx->ofi_cq, &efa_cq_attr);
	if (ret != 0) { NCCL_OFI_WARN("query_cq failed"); MPI_Finalize(); return 1; }

	NCCL_OFI_INFO(NCCL_NET, "Rank %d: SQ entries=%u entry_size=%u CQ entries=%u entry_size=%u",
		      rank, sq_attr.num_entries, sq_attr.entry_size,
		      efa_cq_attr.num_entries, efa_cq_attr.entry_size);

	/* Allocate and register test buffers (host memory for CPU test) */
	const size_t BUF_SIZE = 64;
	const uint8_t PATTERN = 0xAB;
	void *src_buf = calloc(1, BUF_SIZE);
	void *dst_buf = calloc(1, BUF_SIZE);
	if (rank == 0) memset(src_buf, PATTERN, BUF_SIZE);

	struct fid_mr *src_mr = nullptr, *dst_mr = nullptr;
	auto reg_mr = [&](void *buf, struct fid_mr **mr) {
		struct iovec iov = {buf, BUF_SIZE};
		struct fi_mr_attr attr = {};
		attr.mr_iov = &iov;
		attr.iov_count = 1;
		attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
		int r = fi_mr_regattr(ctx->ofi_domain, &attr, 0, mr);
		if (r) { NCCL_OFI_WARN("fi_mr_regattr failed: %s", fi_strerror(-r)); exit(1); }
	};
	reg_mr(src_buf, &src_mr);
	reg_mr(dst_buf, &dst_mr);

	uint32_t src_lkey = (uint32_t)gda_ops->get_mr_lkey(src_mr);
	uint64_t dst_rkey = fi_mr_key(dst_mr);

	/* Allgather rkeys and buffer addresses */
	std::vector<uint64_t> all_rkeys(nranks), all_addrs(nranks);
	all_rkeys[rank] = dst_rkey;
	all_addrs[rank] = (uint64_t)dst_buf;
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_rkeys.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_addrs.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);

	/* Read per-peer addressing from device handle */
	std::vector<uint16_t> h_ahs(nranks), h_qpns(nranks);
	std::vector<uint32_t> h_qkeys(nranks);
	CUDACHECK(cudaMemcpy(h_ahs.data(), h_dev.address_handles, nranks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(h_qpns.data(), h_dev.remote_qpns, nranks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(h_qkeys.data(), h_dev.qkey, nranks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 0: post RDMA write directly to SQ */
	if (rank == 0) {
		int tgt = 1;
		NCCL_OFI_INFO(NCCL_NET, "R0: writing to R%d ah=%u qpn=%u qkey=%u lkey=0x%x rkey=0x%lx",
			      tgt, h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt], src_lkey, all_rkeys[tgt]);

		/* Build WQE */
		struct efa_io_tx_wqe wqe;
		memset(&wqe, 0, sizeof(wqe));

		struct efa_io_tx_meta_desc *meta = &wqe.meta;
		EFA_SET(&meta->ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
		EFA_SET(&meta->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, EFA_IO_RDMA_WRITE);
		EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_PHASE, 0); /* initial phase = 0 */
		EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
		EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
		EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);
		meta->req_id = 0;
		meta->dest_qp_num = h_qpns[tgt];
		meta->ah = h_ahs[tgt];
		meta->qkey = h_qkeys[tgt];
		meta->length = 1; /* 1 SGL entry */

		struct efa_io_remote_mem_addr *remote = &wqe.data.rdma_req.remote_mem;
		remote->rkey = (uint32_t)all_rkeys[tgt];
		remote->buf_addr_lo = (uint32_t)(all_addrs[tgt] & 0xFFFFFFFF);
		remote->buf_addr_hi = (uint32_t)(all_addrs[tgt] >> 32);
		remote->length = BUF_SIZE;

		struct efa_io_tx_buf_desc *local = &wqe.data.rdma_req.local_mem[0];
		local->length = BUF_SIZE;
		EFA_SET(&local->lkey, EFA_IO_TX_BUF_DESC_LKEY, src_lkey);
		local->buf_addr_lo = (uint32_t)((uint64_t)src_buf & 0xFFFFFFFF);
		local->buf_addr_hi = (uint32_t)((uint64_t)src_buf >> 32);

		/* Write WQE to SQ MMIO buffer */
		uint32_t sq_mask = sq_attr.num_entries - 1;
		uint32_t pc = 0; /* first WQE */
		uint32_t sq_offset = (pc & sq_mask) * sizeof(struct efa_io_tx_wqe);
		mmio_memcpy_x64(sq_attr.buffer + sq_offset, &wqe, sizeof(wqe));

		/* Ring doorbell */
		pc = 1;
		mmio_flush_writes();
		mmio_write32(sq_attr.doorbell, pc);
		mmio_flush_writes();

		NCCL_OFI_INFO(NCCL_NET, "R0: WQE posted, polling CQ...");

		/* Poll CQ */
		uint32_t cq_mask = efa_cq_attr.num_entries - 1;
		uint32_t cq_idx = 0;
		volatile struct efa_io_cdesc_common *cqe =
			reinterpret_cast<volatile struct efa_io_cdesc_common *>(
				efa_cq_attr.buffer + (cq_idx & cq_mask) * efa_cq_attr.entry_size);

		NCCL_OFI_INFO(NCCL_NET, "R0: CQ buf=%p entry_size=%u num_entries=%u cqe=%p initial_flags=0x%02x",
			      efa_cq_attr.buffer, efa_cq_attr.entry_size, efa_cq_attr.num_entries,
			      cqe, cqe->flags);

		bool got_completion = false;
		for (int i = 0; i < 100000000; i++) {
			uint8_t flags = *(volatile uint8_t *)&cqe->flags;
			if (FIELD_GET(EFA_IO_CDESC_COMMON_PHASE_MASK, flags) == 1) {
				udma_from_device_barrier();
				NCCL_OFI_INFO(NCCL_NET, "R0: CQ completion status=%u op_type=%lu q_type=%lu req_id=%u",
					      cqe->status,
					      FIELD_GET(EFA_IO_CDESC_COMMON_OP_TYPE_MASK, flags),
					      FIELD_GET(EFA_IO_CDESC_COMMON_Q_TYPE_MASK, flags),
					      cqe->req_id);
				if (cqe->status != EFA_IO_COMP_STATUS_OK) {
					NCCL_OFI_WARN("R0: CQ error status=%u", cqe->status);
				} else if (FIELD_GET(EFA_IO_CDESC_COMMON_Q_TYPE_MASK, flags) != EFA_IO_SEND_QUEUE) {
					NCCL_OFI_WARN("R0: unexpected q_type=%lu", FIELD_GET(EFA_IO_CDESC_COMMON_Q_TYPE_MASK, flags));
				} else {
					got_completion = true;
				}
				break;
			}
		}
		if (!got_completion) {
			NCCL_OFI_WARN("R0: CQ poll timeout. flags=0x%02x", cqe->flags);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 1: verify */
	if (rank == 1) {
		uint8_t *buf = (uint8_t *)dst_buf;
		bool ok = true;
		for (size_t i = 0; i < BUF_SIZE; i++) {
			if (buf[i] != PATTERN) {
				NCCL_OFI_WARN("R1: FAIL byte %zu: 0x%02x != 0x%02x", i, buf[i], PATTERN);
				ok = false;
				break;
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "R1: %s", ok ? "PASS" : "FAIL");
	}

	/* Cleanup */
	fi_close(&dst_mr->fid);
	fi_close(&src_mr->fid);
	free(dst_buf);
	free(src_buf);
#else
	NCCL_OFI_WARN("FI_EFA_GDA_OPS not available, skipping put test");
#endif /* HAVE_DECL_FI_EFA_GDA_OPS */

	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	OFINCCLCHECK(extGin->closeColl(collComm));
	OFINCCLCHECK(extGin->closeListen(listenComm));
	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));
	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);
	return res;
}
