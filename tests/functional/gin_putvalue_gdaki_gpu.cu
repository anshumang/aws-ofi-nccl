/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI PutValue transport validation through GPU-issued RDMA writes.
 *
 * This test constructs WQEs directly; it does not call NCCL's
 * ncclGinApi_PutValue specialization. Backend version 1 stages each value in
 * the plugin-provided source pool and attaches an SGE. Backend version 2 puts
 * the value in the 128-byte WQE's RDMA-write inline region.
 *
 * Each run covers 1-, 2-, 4-, and 8-byte values, both without a signal and
 * with indexed signal 0. Run it once per backend:
 *
 *   mpirun -np 2 gin_putvalue_gdaki_gpu 1
 *   mpirun -np 2 gin_putvalue_gdaki_gpu 2
 *
 * Run with at least 2 MPI ranks on EFA GDA-capable hosts.
 */

#include "config.h"

#include "functional_test.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

#include "efa_cuda_dp_types_v1.h"
#include "efa_cuda_dp_types_v2.h"
#include "efa_cuda_dp_impl.cuh"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int PUTVALUE_SIZE_COUNT = 4;
static constexpr int PUTVALUE_SIGNAL_MODE_COUNT = 2;
static constexpr int PUTVALUE_CASE_COUNT =
	PUTVALUE_SIZE_COUNT * PUTVALUE_SIGNAL_MODE_COUNT;
static constexpr size_t PUTVALUE_DEST_STRIDE = sizeof(uint64_t);
static constexpr size_t PUTVALUE_DEST_BYTES =
	PUTVALUE_CASE_COUNT * PUTVALUE_DEST_STRIDE;
static constexpr int PUTVALUE_MAX_POLL_ITERS = 100000000;
static constexpr uint64_t EFA_COUNTER_MASK = 0x7fffffffULL;

static_assert(sizeof(struct efa_cuda_cq_v1) == sizeof(struct efa_cuda_cq),
	      "backend v1 CQ must retain the common CQ layout");
static_assert(sizeof(struct efa_cuda_cq_v2) == sizeof(struct efa_cuda_cq),
	      "backend v2 CQ must retain the common CQ layout");
static_assert(sizeof(struct efa_cuda_qp_v2) == sizeof(struct efa_cuda_qp),
	      "backend v2 QP must match the vendored device implementation");
static_assert(sizeof(struct efa_io_tx_wqe) == 64,
	      "backend v1 requires a 64-byte WQE");
static_assert(sizeof(struct efa_io_tx_meta_desc) == 32,
	      "backend v2 requires the 32-byte EFA TX metadata descriptor");

/*
 * Backend v1 predates efa_cuda_wr_ctx. Describe its fixed 64-byte WQE so the
 * current builder can encode the frozen layout without duplicating WQE field
 * construction in this test.
 */
__device__ static struct efa_cuda_wr_ctx putvalue_v1_wr_ctx = {
	(uint8_t)offsetof(struct efa_io_tx_wqe,
			  data.rdma_req.remote_mem),
	(uint8_t)offsetof(struct efa_io_tx_wqe,
			  data.rdma_req.local_mem[0]),
	(uint8_t)offsetof(struct efa_io_tx_wqe, data.sgl[0]),
	(uint8_t)offsetof(struct efa_io_tx_wqe, data.inline_data[0]),
	0,
	(uint32_t)sizeof(((struct efa_io_tx_wqe *)nullptr)->data.inline_data),
	1,
	(uint16_t)sizeof(struct efa_io_tx_wqe),
};

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

struct putvalue_kernel_result {
	int32_t error;
	uint32_t completions;
	uint32_t done;
	uint8_t status[PUTVALUE_CASE_COUNT];
	uint8_t q_type[PUTVALUE_CASE_COUNT];
	uint8_t op_type[PUTVALUE_CASE_COUNT];
	uint8_t pad;
	uint16_t req_id[PUTVALUE_CASE_COUNT];
};

struct alignas(uint64_t) putvalue_v2_wqe {
	uint8_t bytes[128];
};

enum putvalue_kernel_error {
	PUTVALUE_KERNEL_SUCCESS = 0,
	PUTVALUE_KERNEL_BAD_V1_STAGING = -1,
	PUTVALUE_KERNEL_BAD_V2_LAYOUT = -2,
	PUTVALUE_KERNEL_WQE_INIT_FAILED = -3,
	PUTVALUE_KERNEL_WQE_POST_FAILED = -4,
	PUTVALUE_KERNEL_CQ_TIMEOUT = -5,
};

__host__ __device__ static inline uint32_t putvalue_size(int size_index)
{
	return 1U << size_index;
}

__host__ __device__ static inline uint64_t putvalue_value(int size_index)
{
	switch (size_index) {
	case 0:
		return 0xA5ULL;
	case 1:
		return 0xB6C7ULL;
	case 2:
		return 0xD8E9FA0BULL;
	default:
		return 0x1122334455667788ULL;
	}
}

__host__ __device__ static inline int putvalue_case_index(bool signaled,
							   int size_index)
{
	return (signaled ? PUTVALUE_SIZE_COUNT : 0) + size_index;
}

__host__ __device__ static inline uint64_t putvalue_dest_offset(bool signaled,
								 int size_index)
{
	return (uint64_t)putvalue_case_index(signaled, size_index) *
		PUTVALUE_DEST_STRIDE;
}

__device__ static int post_putvalue_v1(
	struct nccl_ofi_gin_gdaki_dev_handle *dev, int peer, uint64_t dst_addr,
	uint32_t dst_rkey, uint64_t value, uint32_t value_bytes, bool signaled,
	uint16_t wr_id)
{
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle *ep = &dev->pvdata;
	auto *qp_v1 = reinterpret_cast<struct efa_cuda_qp_v1 *>(ep->qp);

	if (ep->putvalue_slice_base == 0 || dev->putvalue_lkey == 0 ||
	    dev->putvalue_slot_size < value_bytes || ep->sq_size == 0) {
		return PUTVALUE_KERNEL_BAD_V1_STAGING;
	}

	uint64_t slot = (uint64_t)qp_v1->sq.wq.pc % (uint64_t)ep->sq_size;
	uint64_t staging_addr = ep->putvalue_slice_base +
		slot * (uint64_t)dev->putvalue_slot_size;
	const uint8_t *value_bytes_ptr = reinterpret_cast<const uint8_t *>(&value);
	uint8_t *staging_ptr = reinterpret_cast<uint8_t *>(staging_addr);
	for (uint32_t i = 0; i < value_bytes; i++) {
		staging_ptr[i] = value_bytes_ptr[i];
	}

	struct efa_io_tx_wqe wr;
	EfaCudaWrBuilder wr_builder(
		&putvalue_v1_wr_ctx, reinterpret_cast<uint8_t *>(&wr));
	int ret = wr_builder.init_rdma_write(
		(uint64_t)wr_id, dst_rkey, dst_addr);
	if (ret != 0) {
		return PUTVALUE_KERNEL_WQE_INIT_FAILED;
	}

	ret = wr_builder.set_sge(
		dev->putvalue_lkey, staging_addr, value_bytes);
	if (ret != 0) {
		return PUTVALUE_KERNEL_WQE_INIT_FAILED;
	}

	const uint32_t target_slot = signaled ? 1U : 0U;
	const uint32_t target_index =
		target_slot * (uint32_t)dev->nranks + (uint32_t)peer;
	wr_builder.set_remote(
		ep->target_address_handles[target_index],
		(uint32_t)ep->target_remote_qpns[target_index],
		ep->target_qkey[target_index]);
	wr_builder.set_processing_hints(
		EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE);

	struct efa_cuda_wq_v1 *wq = &qp_v1->sq.wq;
	uint32_t pc = wq->pc;
	uint32_t sq_index = pc & wq->queue_mask;
	int wqe_phase = wq->phase ^
		(int)((pc & wq->queue_mask) >> wq->queue_size_shift);
	EFA_SET(&wr.meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase);

	uint64_t *src = reinterpret_cast<uint64_t *>(&wr);
	uint64_t *dst = reinterpret_cast<uint64_t *>(
		wq->buf + (uint64_t)sq_index * sizeof(wr));
	for (uint32_t i = 0; i < sizeof(wr) / sizeof(uint64_t); i++) {
		dst[i] = src[i];
	}

	wq->phase ^= (int)(((pc & wq->queue_mask) + 1U) >>
			  wq->queue_size_shift);
	wq->pc = pc + 1U;
	__threadfence_system();
	*wq->db = wq->pc;
	__threadfence_system();
	return PUTVALUE_KERNEL_SUCCESS;
}

__device__ static int post_putvalue_v2(
	struct nccl_ofi_gin_gdaki_dev_handle *dev, int peer, uint64_t dst_addr,
	uint32_t dst_rkey, uint64_t value, uint32_t value_bytes, bool signaled,
	uint16_t wr_id)
{
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle *ep = &dev->pvdata;
	auto *qp = reinterpret_cast<struct efa_cuda_qp_v2 *>(ep->qp);
	struct efa_cuda_wr_ctx_v2 *wr_ctx = &qp->sq.wr_ctx;

	if (wr_ctx->wqe_size != sizeof(struct putvalue_v2_wqe) ||
	    wr_ctx->write_inline_data_offset == 0 ||
	    wr_ctx->max_inline_data < value_bytes ||
	    (uint32_t)wr_ctx->write_inline_data_offset + value_bytes >
		    wr_ctx->wqe_size) {
		return PUTVALUE_KERNEL_BAD_V2_LAYOUT;
	}

	struct putvalue_v2_wqe wr;
	EfaCudaWrBuilder wr_builder(
		reinterpret_cast<struct efa_cuda_wr_ctx *>(wr_ctx), wr.bytes);
	int ret = wr_builder.init_rdma_write(
		(uint64_t)wr_id, dst_rkey, dst_addr);
	if (ret != 0 ||
	    wr_builder.set_inline_data(&value, value_bytes) != 0) {
		return PUTVALUE_KERNEL_WQE_INIT_FAILED;
	}

	const uint32_t target_slot = signaled ? 1U : 0U;
	const uint32_t target_index =
		target_slot * (uint32_t)dev->nranks + (uint32_t)peer;
	wr_builder.set_remote(
		ep->target_address_handles[target_index],
		(uint32_t)ep->target_remote_qpns[target_index],
		ep->target_qkey[target_index]);
	wr_builder.set_processing_hints(
		EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE);

	ret = efa_cuda_start_sq_batch(
		reinterpret_cast<struct efa_cuda_qp *>(qp), 1);
	if (ret != 0 ||
	    efa_cuda_sq_batch_place_wr(
		    reinterpret_cast<struct efa_cuda_qp *>(qp), 0, &wr) != 0) {
		return PUTVALUE_KERNEL_WQE_POST_FAILED;
	}
	efa_cuda_flush_sq_wrs(reinterpret_cast<struct efa_cuda_qp *>(qp));

	return PUTVALUE_KERNEL_SUCCESS;
}

__device__ static int poll_putvalue_completion(
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle *ep, int case_index,
	struct putvalue_kernel_result *result)
{
	auto *cq = reinterpret_cast<struct efa_cuda_cq *>(ep->cq);

	for (int i = 0; i < PUTVALUE_MAX_POLL_ITERS; i++) {
		void *wc = efa_cuda_cq_poll(cq, 0);
		if (wc == nullptr) {
			continue;
		}

		auto *cqe = reinterpret_cast<struct efa_io_cdesc_common *>(wc);
		result->status[case_index] = cqe->status;
		result->q_type[case_index] =
			(uint8_t)((cqe->flags >> 1) & 0x3);
		result->op_type[case_index] =
			(uint8_t)efa_cuda_wc_read_opcode(wc);
		result->req_id[case_index] = efa_cuda_wc_read_req_id(wc);
		efa_cuda_cq_pop(cq, 1);
		result->completions++;
		return PUTVALUE_KERNEL_SUCCESS;
	}

	return PUTVALUE_KERNEL_CQ_TIMEOUT;
}

__global__ void gin_putvalue_gpu_kernel(
	struct nccl_ofi_gin_gdaki_dev_handle *dev, int backend_version, int peer,
	uint64_t dst_addr, uint32_t dst_rkey,
	struct putvalue_kernel_result *result)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}

	for (int signal_mode = 0; signal_mode < PUTVALUE_SIGNAL_MODE_COUNT;
	     signal_mode++) {
		const bool signaled = signal_mode != 0;
		for (int size_index = 0; size_index < PUTVALUE_SIZE_COUNT;
		     size_index++) {
			const int case_index =
				putvalue_case_index(signaled, size_index);
			const uint32_t value_bytes = putvalue_size(size_index);
			const uint64_t value = putvalue_value(size_index);
			const uint64_t case_dst_addr = dst_addr +
				putvalue_dest_offset(signaled, size_index);

			int ret;
			if (backend_version == 1) {
				ret = post_putvalue_v1(
					dev, peer, case_dst_addr, dst_rkey,
					value, value_bytes, signaled,
					(uint16_t)case_index);
			} else {
				ret = post_putvalue_v2(
					dev, peer, case_dst_addr, dst_rkey,
					value, value_bytes, signaled,
					(uint16_t)case_index);
			}
			if (ret != PUTVALUE_KERNEL_SUCCESS) {
				result->error = ret;
				return;
			}

			ret = poll_putvalue_completion(&dev->pvdata, case_index,
						       result);
			if (ret != PUTVALUE_KERNEL_SUCCESS) {
				result->error = ret;
				return;
			}
		}
	}

	result->done = 1;
}

static int parse_backend_version(int argc, char *argv[])
{
	if (argc != 2) {
		return NCCL_OFI_GDAKI_BACKEND_VERSION_UNSET;
	}

	char *end = nullptr;
	long version = strtol(argv[1], &end, 10);
	if (end == argv[1] || *end != '\0' ||
	    (version != 1 && version != 2)) {
		return NCCL_OFI_GDAKI_BACKEND_VERSION_UNSET;
	}

	return (int)version;
}

static uint64_t counter_delta(uint64_t before, uint64_t after)
{
	return (after - before) & EFA_COUNTER_MASK;
}

int main(int argc, char *argv[])
{
	int rank, nranks, proc_name_len, local_rank = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	const int backend_version = parse_backend_version(argc, argv);
	if (backend_version == NCCL_OFI_GDAKI_BACKEND_VERSION_UNSET) {
		if (rank == 0) {
			fprintf(stderr,
				"Usage: %s <backend-version: 1|2>\n", argv[0]);
		}
		MPI_Finalize();
		return ncclInvalidArgument;
	}

	if (nranks < 2) {
		NCCL_OFI_WARN("Need at least 2 ranks");
		MPI_Finalize();
		return ncclInvalidArgument;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)],
			       &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		      all_proc_name.data(), MPI_MAX_PROCESSOR_NAME, MPI_BYTE,
		      MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)]) &&
		    i < rank) {
			local_rank++;
		}
	}
	CUDACHECK(cudaSetDevice(local_rank));

	set_system_page_size();
	void *net_plugin_handle = load_netPlugin();
	test_nccl_net_t *ext_net = get_netPlugin_symbol(net_plugin_handle);
	test_nccl_gin_t *ext_gin = get_ginPlugin_symbol(net_plugin_handle);
	if (ext_net == nullptr || ext_gin == nullptr) {
		MPI_Finalize();
		return ncclInternalError;
	}

	void *net_ctx = nullptr;
	ncclNetCommConfig_v11_t net_config = {};
	OFINCCLCHECK(ext_net->init(&net_ctx, 0, &net_config,
				   &functional_test_logger, nullptr));

	void *gin_ctx = nullptr;
	OFINCCLCHECK(ext_gin->init(&gin_ctx, 0, &functional_test_logger));

	int ndev;
	OFINCCLCHECK(ext_gin->devices(&ndev));
	int dev = local_rank % ndev;

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handle_ptrs(nranks);
	void *listen_comm = nullptr;
	OFINCCLCHECK(ext_gin->listen(gin_ctx, dev, handles[rank].handle,
				     &listen_comm));

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(),
		      NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) {
		handle_ptrs[i] = &handles[i];
	}

	void *coll_comm = nullptr;
	OFINCCLCHECK(ext_gin->connect(gin_ctx, handle_ptrs.data(), nranks,
				      rank, listen_comm, &coll_comm));

	test_nccl_gin_config_t gin_config = {};
	gin_config.nSignals = 1;
	gin_config.nContexts = 1;
	gin_config.queueDepth = 64;
	gin_config.trafficClass = -1;
	gin_config.backendVersion = backend_version;

	void *proxy_ctx = nullptr;
	ncclNetDeviceHandle_v11_t *dev_handle = nullptr;
	OFINCCLCHECK(ext_gin->createContext(coll_comm, &gin_config, &proxy_ctx,
					    &dev_handle));
	NCCL_OFI_INFO(NCCL_NET,
		      "Rank %d: createContext done (backendVersion=%d)",
		      rank, backend_version);

	auto *dev_gpu = reinterpret_cast<struct nccl_ofi_gin_gdaki_dev_handle *>(
		dev_handle->handle);
	struct nccl_ofi_gin_gdaki_dev_handle dev_host = {};
	CUDACHECK(cudaMemcpy(&dev_host, dev_gpu, sizeof(dev_host),
			     cudaMemcpyDeviceToHost));

	int local_ready =
		dev_host.nSignals == 1 && dev_host.signal_handles != nullptr &&
		dev_host.pvdata.qp != nullptr && dev_host.pvdata.cq != nullptr &&
		dev_host.pvdata.target_address_handles != nullptr &&
		dev_host.pvdata.target_remote_qpns != nullptr &&
		dev_host.pvdata.target_qkey != nullptr &&
		dev_host.pvdata.local_cntr_value != nullptr &&
		dev_host.pvdata.sq_size != 0;

	struct nccl_ofi_gin_gdaki_dev_counter_handle *signal_gpu = nullptr;
	struct nccl_ofi_gin_gdaki_dev_counter_handle signal_host = {};
	if (local_ready) {
		CUDACHECK(cudaMemcpy(&signal_gpu, dev_host.signal_handles,
				     sizeof(signal_gpu), cudaMemcpyDeviceToHost));
		local_ready = signal_gpu != nullptr;
	}
	if (local_ready) {
		CUDACHECK(cudaMemcpy(&signal_host, signal_gpu,
				     sizeof(signal_host), cudaMemcpyDeviceToHost));
		local_ready = signal_host.cntr_value != nullptr;
	}

	if (local_ready && backend_version == 1) {
		local_ready = dev_host.pvdata.putvalue_slice_base != 0 &&
			dev_host.putvalue_lkey != 0 &&
			dev_host.putvalue_slot_size == sizeof(uint64_t);

		struct efa_cuda_qp_v1 qp_host = {};
		CUDACHECK(cudaMemcpy(&qp_host, dev_host.pvdata.qp,
				     sizeof(qp_host), cudaMemcpyDeviceToHost));
		local_ready = local_ready &&
			qp_host.sq.wq.max_wqes == dev_host.pvdata.sq_size;
	} else if (local_ready) {
		local_ready = dev_host.pvdata.putvalue_slice_base == 0 &&
			dev_host.putvalue_lkey == 0 &&
			dev_host.putvalue_slot_size == 0;

		struct efa_cuda_qp_v2 qp_host = {};
		CUDACHECK(cudaMemcpy(&qp_host, dev_host.pvdata.qp,
				     sizeof(qp_host), cudaMemcpyDeviceToHost));
		local_ready = local_ready &&
			qp_host.sq.wr_ctx.wqe_size ==
				sizeof(struct putvalue_v2_wqe) &&
			qp_host.sq.wr_ctx.write_inline_data_offset != 0 &&
			qp_host.sq.wr_ctx.max_inline_data >= sizeof(uint64_t);
	}

	int all_ready = 0;
	MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN,
		      MPI_COMM_WORLD);
	if (!all_ready) {
		NCCL_OFI_WARN(
			"Rank %d: backend %d device handle failed PutValue preflight",
			rank, backend_version);
		(void)ext_gin->destroyContext(proxy_ctx);
		(void)ext_gin->closeColl(coll_comm);
		(void)ext_gin->closeListen(listen_comm);
		(void)ext_gin->finalize(gin_ctx);
		(void)ext_net->finalize(net_ctx);
		dlclose(net_plugin_handle);
		MPI_Finalize();
		return ncclInternalError;
	}

	void *dst_gpu = nullptr;
	CUDACHECK(cudaMalloc(&dst_gpu, PUTVALUE_DEST_BYTES));
	CUDACHECK(cudaMemset(dst_gpu, 0, PUTVALUE_DEST_BYTES));

	void *dst_mhandle = nullptr;
	void *dst_ginhandle = nullptr;
	OFINCCLCHECK(ext_gin->regMrSym(coll_comm, dst_gpu,
				       PUTVALUE_DEST_BYTES, NCCL_PTR_CUDA, 0,
				       &dst_mhandle, &dst_ginhandle));
	if (dst_ginhandle == nullptr) {
		NCCL_OFI_WARN("regMrSym returned a null GIN handle");
		MPI_Finalize();
		return ncclInternalError;
	}

	const size_t mr_handle_bytes =
		sizeof(struct nccl_ofi_gin_gdaki_mr_handle) +
		(size_t)nranks *
			sizeof(struct nccl_ofi_gin_gdaki_mr_peer);
	auto **rail_handles = reinterpret_cast<
		struct nccl_ofi_gin_gdaki_mr_handle **>(dst_ginhandle);
	struct nccl_ofi_gin_gdaki_mr_handle *dst_rail_gpu = nullptr;
	CUDACHECK(cudaMemcpy(&dst_rail_gpu,
			     rail_handles + dev_host.rail_id,
			     sizeof(dst_rail_gpu), cudaMemcpyDeviceToHost));

	std::vector<uint8_t> dst_mr_storage(mr_handle_bytes);
	CUDACHECK(cudaMemcpy(dst_mr_storage.data(), dst_rail_gpu,
			     mr_handle_bytes, cudaMemcpyDeviceToHost));
	auto *dst_mr = reinterpret_cast<
		struct nccl_ofi_gin_gdaki_mr_handle *>(
		dst_mr_storage.data());

	std::vector<uint32_t> all_rkeys(nranks);
	for (int i = 0; i < nranks; i++) {
		all_rkeys[i] = dst_mr->peers[i].rkey;
	}

	std::vector<uint64_t> all_dst_addrs(nranks);
	uint64_t local_dst_addr = (uint64_t)dst_gpu;
	MPI_Allgather(&local_dst_addr, 1, MPI_UINT64_T,
		      all_dst_addrs.data(), 1, MPI_UINT64_T,
		      MPI_COMM_WORLD);

	uint64_t write_counter_before = 0;
	uint64_t signal_counter_before = 0;
	CUDACHECK(cudaMemcpy(
		&write_counter_before,
		(void *)dev_host.pvdata.local_cntr_value,
		sizeof(write_counter_before), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&signal_counter_before,
			     (void *)signal_host.cntr_value,
			     sizeof(signal_counter_before),
			     cudaMemcpyDeviceToHost));

	MPI_Barrier(MPI_COMM_WORLD);

	int local_pass = 1;
	if (rank == 0) {
		const int peer = 1;
		struct putvalue_kernel_result *result_gpu = nullptr;
		CUDACHECK(cudaMalloc(&result_gpu, sizeof(*result_gpu)));
		CUDACHECK(cudaMemset(result_gpu, 0, sizeof(*result_gpu)));

		gin_putvalue_gpu_kernel<<<1, 1>>>(
			dev_gpu, backend_version, peer, all_dst_addrs[peer],
			all_rkeys[peer], result_gpu);
		CUDACHECK(cudaGetLastError());
		CUDACHECK(cudaDeviceSynchronize());

		struct putvalue_kernel_result result_host = {};
		CUDACHECK(cudaMemcpy(&result_host, result_gpu,
				     sizeof(result_host),
				     cudaMemcpyDeviceToHost));
		CUDACHECK(cudaFree(result_gpu));

		if (!result_host.done ||
		    result_host.error != PUTVALUE_KERNEL_SUCCESS ||
		    result_host.completions != (uint32_t)PUTVALUE_CASE_COUNT) {
			NCCL_OFI_WARN(
				"R0: kernel failed: done=%u error=%d completions=%u",
				result_host.done, result_host.error,
				result_host.completions);
			local_pass = 0;
		}

		for (int i = 0; i < PUTVALUE_CASE_COUNT; i++) {
			bool completion_ok =
				result_host.status[i] == EFA_IO_COMP_STATUS_OK &&
				result_host.q_type[i] == EFA_IO_SEND_QUEUE &&
				result_host.op_type[i] ==
					EFA_CUDA_WC_RDMA_WRITE &&
				result_host.req_id[i] == (uint16_t)i;
			if (!completion_ok) {
				NCCL_OFI_WARN(
					"R0: case %d completion status=%u q_type=%u op_type=%u req_id=%u",
					i, (unsigned int)result_host.status[i],
					(unsigned int)result_host.q_type[i],
					(unsigned int)result_host.op_type[i],
					(unsigned int)result_host.req_id[i]);
				local_pass = 0;
			}
		}
	}

	int posting_pass = local_pass;
	MPI_Bcast(&posting_pass, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 1 && posting_pass) {
		uint64_t signal_counter = signal_counter_before;
		for (int i = 0; i < PUTVALUE_MAX_POLL_ITERS; i++) {
			CUDACHECK(cudaMemcpy(&signal_counter,
					     (void *)signal_host.cntr_value,
					     sizeof(signal_counter),
					     cudaMemcpyDeviceToHost));
			if (counter_delta(signal_counter_before,
					  signal_counter) >=
			    PUTVALUE_SIZE_COUNT) {
				break;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	uint64_t write_counter_after = 0;
	uint64_t signal_counter_after = 0;
	CUDACHECK(cudaMemcpy(
		&write_counter_after,
		(void *)dev_host.pvdata.local_cntr_value,
		sizeof(write_counter_after), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&signal_counter_after,
			     (void *)signal_host.cntr_value,
			     sizeof(signal_counter_after),
			     cudaMemcpyDeviceToHost));

	if (rank == 0) {
		uint64_t delta =
			counter_delta(write_counter_before, write_counter_after);
		if (delta != (uint64_t)PUTVALUE_CASE_COUNT) {
			NCCL_OFI_WARN(
				"R0: FI_WRITE counter delta=%lu, expected=%d",
				(unsigned long)delta, PUTVALUE_CASE_COUNT);
			local_pass = 0;
		}
	} else if (rank == 1) {
		uint64_t delta = counter_delta(signal_counter_before,
					       signal_counter_after);
		if (delta != (uint64_t)PUTVALUE_SIZE_COUNT) {
			NCCL_OFI_WARN(
				"R1: FI_REMOTE_WRITE counter delta=%lu, expected=%d",
				(unsigned long)delta, PUTVALUE_SIZE_COUNT);
			local_pass = 0;
		}

		std::array<uint8_t, PUTVALUE_DEST_BYTES> expected = {};
		for (int signal_mode = 0;
		     signal_mode < PUTVALUE_SIGNAL_MODE_COUNT;
		     signal_mode++) {
			bool signaled = signal_mode != 0;
			for (int size_index = 0;
			     size_index < PUTVALUE_SIZE_COUNT;
			     size_index++) {
				uint64_t value = putvalue_value(size_index);
				uint32_t value_bytes =
					putvalue_size(size_index);
				size_t offset = (size_t)putvalue_dest_offset(
					signaled, size_index);
				for (uint32_t i = 0; i < value_bytes; i++) {
					expected[offset + i] =
						(uint8_t)(value >> (8U * i));
				}
			}
		}

		std::array<uint8_t, PUTVALUE_DEST_BYTES> actual = {};
		CUDACHECK(cudaMemcpy(actual.data(), dst_gpu, actual.size(),
				     cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < actual.size(); i++) {
			if (actual[i] != expected[i]) {
				NCCL_OFI_WARN(
					"R1: destination byte %zu is 0x%02x, expected 0x%02x",
					i, (unsigned int)actual[i],
					(unsigned int)expected[i]);
				local_pass = 0;
				break;
			}
		}
	}

	int global_pass = 0;
	MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN,
		      MPI_COMM_WORLD);

	OFINCCLCHECK(ext_gin->deregMrSym(coll_comm, dst_mhandle));
	CUDACHECK(cudaFree(dst_gpu));
	OFINCCLCHECK(ext_gin->destroyContext(proxy_ctx));
	OFINCCLCHECK(ext_gin->closeColl(coll_comm));
	OFINCCLCHECK(ext_gin->closeListen(listen_comm));
	OFINCCLCHECK(ext_gin->finalize(gin_ctx));
	OFINCCLCHECK(ext_net->finalize(net_ctx));
	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET,
		      "Rank %d: backend %d PutValue transport test %s",
		      rank, backend_version, global_pass ? "PASS" : "FAIL");
	return global_pass ? ncclSuccess : ncclSystemError;
}
