/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Minimal functional test for the GIN createContext / destroyContext pair.
 *
 * Exercises the full connect() path up through createContext, then tears
 * everything down. No iput / iputSignal calls, so this test passes in both
 * proxy mode (OFI_NCCL_GIN_GDAKI unset or =0) and GDAKI mode
 * (OFI_NCCL_GIN_GDAKI=1).
 *
 * Run with at least 2 MPI ranks.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <vector>

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
		NCCL_OFI_WARN("Expected at least two ranks but got %d. "
			      "gin_createcontext should be run with at least two ranks.",
			      nranks);
		MPI_Finalize();
		return ncclInvalidArgument;
	}

	/* Determine local rank so we can pick a CUDA device. */
	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", local_rank);
	CUDACHECK(cudaSetDevice(local_rank));

	/* Get external Network from NCCL-OFI library */
	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (extNet == nullptr || extGin == nullptr) {
		NCCL_OFI_WARN("Failed to load net or gin plugin symbols");
		MPI_Finalize();
		return ncclInternalError;
	}

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	/*
	 * Net plugin must be initialized before gin plugin even when the test
	 * does not itself use the net plugin; gin init relies on shared
	 * structures created by net init.
	 */
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx_plugin = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx_plugin, 0, &functional_test_logger));
	NCCL_OFI_INFO(NCCL_NET,
		      "Process rank %d started. NCCL-GIN device used on %s is %s.", rank,
		      &all_proc_name[PROC_NAME_IDX(rank)], extGin->name);

	int ndev;
	OFINCCLCHECK(extGin->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);
	if (ndev == 0) {
		NCCL_OFI_WARN("No GIN devices available");
		MPI_Finalize();
		return ncclInternalError;
	}

	int dev = local_rank % ndev;

	ncclNetProperties_v12_t props = {};
	OFINCCLCHECK(extGin->getProperties(dev, &props));
	if (!is_gdr_supported_nic(props.ptrSupport)) {
		NCCL_OFI_WARN("Device %d does not report NCCL_PTR_CUDA", dev);
	}

	/* listen + allgather of connect handles. */
	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx_plugin, dev, handles[rank].handle, &listenComm));
	assert(listenComm);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(),
		      NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx_plugin, handles_ptrs.data(), nranks, rank, listenComm,
				     &collComm));
	assert(collComm != nullptr);

	/* createContext / destroyContext: the core of what this test validates. */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 0;
	ginConfig.nCounters = 0;
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	assert(proxyCtx != nullptr);

	if (devHandle != nullptr) {
		NCCL_OFI_INFO(NCCL_NET,
			      "Rank %d: createContext produced device handle. "
			      "netDeviceType=%d handle=%p size=%zu needsProxyProgress=%d",
			      rank, (int)devHandle->netDeviceType, devHandle->handle,
			      devHandle->size, devHandle->needsProxyProgress);
	} else {
		NCCL_OFI_INFO(NCCL_NET,
			      "Rank %d: createContext returned devHandle=NULL (proxy mode)", rank);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Tear down in reverse order. */
	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	proxyCtx = nullptr;

	OFINCCLCHECK(extGin->closeColl(collComm));
	collComm = nullptr;
	OFINCCLCHECK(extGin->closeListen(listenComm));
	listenComm = nullptr;

	OFINCCLCHECK(extGin->finalize(ginCtx_plugin));
	OFINCCLCHECK(extNet->finalize(netCtx));

	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return res;
}
