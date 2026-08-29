// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

/*
 * Unit tests for the host-side queue initialization API. Host-only: no CUDA
 * toolkit, no GPU. Build and run via `make test`.
 */

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "../host/efa_cuda_dp.h"
#include "../common/efa_cuda_dp_types.h"

static int pass;
static int fail;

#define CHECK(cond, msg)                                                                           \
	do {                                                                                       \
		if (cond) {                                                                        \
			pass++;                                                                    \
		} else {                                                                           \
			fail++;                                                                    \
			printf("    failed check: %s\n", msg);                                               \
		}                                                                                  \
	} while (0)

/* Valid baseline attributes the tests perturb. */
static uint8_t cq_buf[4096];
static uint8_t sq_buf[8192];
static uint8_t rq_buf[8192];
static uint32_t sq_db;
static uint32_t rq_db;

static struct efa_cuda_cq_attrs valid_cq_attrs(void)
{
	struct efa_cuda_cq_attrs attrs;

	memset(&attrs, 0, sizeof(attrs));
	attrs.buffer = cq_buf;
	attrs.num_entries = 64;
	attrs.entry_size = 16;

	return attrs;
}

static struct efa_cuda_qp_attrs valid_qp_attrs(uint32_t sq_entry_size, uint32_t sq_caps)
{
	struct efa_cuda_qp_attrs attrs;

	memset(&attrs, 0, sizeof(attrs));
	attrs.sq_buffer = sq_buf;
	attrs.rq_buffer = rq_buf;
	attrs.sq_doorbell = &sq_db;
	attrs.rq_doorbell = &rq_db;
	attrs.sq_num_entries = 128;
	attrs.rq_num_entries = 32;
	attrs.sq_max_batch = 8;
	attrs.sq_entry_size = sq_entry_size;
	attrs.rq_entry_size = 16;
	attrs.sq_caps = sq_caps;

	return attrs;
}

static void test_context_lifecycle(struct efa_cuda_host_context *v0,
				   struct efa_cuda_host_context *v1)
{
	CHECK(v0, "context created for major 0");
	CHECK(v1, "context created for major 1");
	CHECK(efa_cuda_host_context_create(99, 0, 0) == nullptr, "unsupported major rejected");
	CHECK(efa_cuda_init_cq(nullptr, nullptr, 0, nullptr, 0) == -EINVAL,
	      "NULL context rejected by init_cq");
	CHECK(efa_cuda_init_qp(nullptr, nullptr, 0, nullptr, 0) == -EINVAL,
	      "NULL context rejected by init_qp");
}

static void test_init_cq(struct efa_cuda_host_context *v0, struct efa_cuda_host_context *v1)
{
	struct efa_cuda_cq_attrs attrs = valid_cq_attrs();
	struct efa_cuda_cq_v0 cq_from_v0, cq_from_v1;

	memset(&cq_from_v1, 0xAA, sizeof(cq_from_v1));
	CHECK(efa_cuda_init_cq(v1, &cq_from_v1, sizeof(cq_from_v1), &attrs, sizeof(attrs)) == 0,
	      "init_cq via major 1");
	CHECK(cq_from_v1.num_entries == 64 && cq_from_v1.queue_mask == 63 &&
		      cq_from_v1.queue_size_shift == 6 && cq_from_v1.phase == 1,
	      "CQ fields derived from num_entries");

	memset(&cq_from_v0, 0xAA, sizeof(cq_from_v0));
	CHECK(efa_cuda_init_cq(v0, &cq_from_v0, sizeof(cq_from_v0), &attrs, sizeof(attrs)) == 0,
	      "init_cq via major 0");
	CHECK(memcmp(&cq_from_v0, &cq_from_v1, sizeof(cq_from_v0)) == 0,
	      "both majors produce byte-identical CQs");

	attrs.num_entries = 63;
	CHECK(efa_cuda_init_cq(v1, &cq_from_v1, sizeof(cq_from_v1), &attrs, sizeof(attrs)) ==
		      -EINVAL,
	      "non-power-of-2 CQ size rejected");
}

static void test_init_qp_v1(struct efa_cuda_host_context *, struct efa_cuda_host_context *v1)
{
	struct efa_cuda_qp_attrs attrs =
		valid_qp_attrs(128, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_v1 qp;

	attrs.sq_max_inline_data = 80;
	attrs.sq_max_rdma_sges = 1;
	memset(&qp, 0xAA, sizeof(qp));
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0,
	      "init_qp with the 128B WQE");
	CHECK(qp.sq.wr_ctx.wqe_size == 128 && qp.sq.wr_ctx.max_inline_data == 80,
	      "wr_ctx records WQE size and inline limit");
	CHECK(qp.sq.wr_ctx.write_inline_data_offset != 0,
	      "128B WQE supports rdma-write inline");
	CHECK(qp.sq.wq.queue_mask == 127 && qp.sq.wq.queue_size_shift == 7,
	      "SQ mask and shift derived from num_entries");
	CHECK(qp.rq.wq.phase == 1 && qp.sq.wq.phase == 0, "SQ phase 0, RQ phase 1");

	attrs.sq_entry_size = 64;
	attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0,
	      "init_qp with the 64B WQE");
	CHECK(qp.sq.wr_ctx.write_inline_data_offset == 0,
	      "64B WQE has no rdma-write inline");

	attrs.sq_caps = 0;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "64-bit request ID capability required");

	attrs.sq_caps = EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID;
	attrs.sq_entry_size = 96;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "invalid WQE size rejected");

	attrs.sq_entry_size = 64;
	attrs.rq_caps = 0x2;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "RQ capabilities rejected");
}

static void test_init_qp_v0(struct efa_cuda_host_context *v0, struct efa_cuda_host_context *)
{
	struct efa_cuda_qp_attrs attrs = valid_qp_attrs(64, 0);
	struct efa_cuda_qp_v0 qp;

	memset(&qp, 0xAA, sizeof(qp));
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0, "init_qp");
	CHECK(qp.sq.max_inline_data == 0 && qp.sq.max_rdma_sges == 0,
	      "zero limits passed through");

	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 1;
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0,
	      "caller limits accepted");
	CHECK(qp.sq.max_inline_data == 32 && qp.sq.max_rdma_sges == 1,
	      "caller limits passed through");

	attrs.sq_max_inline_data = 0;
	attrs.sq_max_rdma_sges = 0;
	attrs.sq_entry_size = 128;
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "128B WQE rejected");

	attrs.sq_entry_size = 64;
	attrs.sq_caps = EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID;
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "capabilities rejected");
}

static void test_storage_size_mismatch(struct efa_cuda_host_context *v0,
				       struct efa_cuda_host_context *v1)
{
	struct efa_cuda_qp_attrs attrs_v1 =
		valid_qp_attrs(64, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_attrs attrs_v0 = valid_qp_attrs(64, 0);
	struct efa_cuda_qp_v1 qp1;
	uint8_t big_buf[sizeof(struct efa_cuda_qp_v1)];

	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(struct efa_cuda_qp_v0), &attrs_v1,
			       sizeof(attrs_v1)) == -EINVAL,
	      "undersized storage rejected");
	CHECK(efa_cuda_init_qp(v0, big_buf, sizeof(big_buf), &attrs_v0,
			       sizeof(attrs_v0)) == 0,
	      "oversized storage accepted");
}

static void test_attrs_forward_compat(struct efa_cuda_host_context *, struct efa_cuda_host_context *v1)
{
	struct bigger_attrs {
		struct efa_cuda_qp_attrs attrs;
		uint32_t future_field;
	} big;
	struct efa_cuda_qp_v1 qp;

	memset(&big, 0, sizeof(big));
	big.attrs = valid_qp_attrs(64, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	big.attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &big.attrs, sizeof(big)) == 0,
	      "larger attrs with zeroed tail accepted");

	big.future_field = 7;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &big.attrs, sizeof(big)) == -EINVAL,
	      "larger attrs with set tail rejected");
}

static void test_sq_limits(struct efa_cuda_host_context *v0, struct efa_cuda_host_context *v1)
{
	struct efa_cuda_qp_attrs attrs =
		valid_qp_attrs(128, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_v1 qp1;
	struct efa_cuda_qp_v0 qp0;

	attrs.sq_max_rdma_sges = 1;
	attrs.sq_max_inline_data = 80;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == 0,
	      "128B WQE: inline 80 accepted");
	attrs.sq_max_inline_data = 81;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "128B WQE: inline 81 rejected");
	attrs.sq_max_inline_data = 200;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "128B WQE: inline 200 rejected");

	attrs.sq_entry_size = 64;
	attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == 0,
	      "64B WQE: inline 32 accepted");
	attrs.sq_max_inline_data = 33;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "64B WQE: inline 33 rejected");
	attrs.sq_max_inline_data = 80;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "64B WQE: inline 80 rejected");
	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 2;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "major 1: 2 RDMA SGEs rejected");

	attrs = valid_qp_attrs(64, 0);
	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 1;
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), &attrs, sizeof(attrs)) == 0,
	      "major 0, 64B WQE: inline 32 accepted");
	attrs.sq_max_inline_data = 80;
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), &attrs, sizeof(attrs)) == -EINVAL,
	      "major 0, 64B WQE: inline 80 rejected");
	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 2;
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), &attrs, sizeof(attrs)) == -EINVAL,
	      "major 0: 2 RDMA SGEs rejected");
}

static void test_size_queries(struct efa_cuda_host_context *v0, struct efa_cuda_host_context *v1)
{
	struct efa_cuda_qp_attrs attrs =
		valid_qp_attrs(64, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_v1 qp;

	attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_get_cq_size(v0) == (int)sizeof(struct efa_cuda_cq_v0),
	      "CQ size, major 0");
	CHECK(efa_cuda_get_cq_size(v1) == (int)sizeof(struct efa_cuda_cq_v0),
	      "CQ size, major 1");
	CHECK(efa_cuda_get_qp_size(v0) == (int)sizeof(struct efa_cuda_qp_v0),
	      "QP size, major 0");
	CHECK(efa_cuda_get_qp_size(v1) == (int)sizeof(struct efa_cuda_qp_v1),
	      "QP size, major 1");
	CHECK(efa_cuda_get_qp_size(nullptr) == -EINVAL, "QP size, NULL context");
	CHECK(efa_cuda_init_qp(v1, &qp, efa_cuda_get_qp_size(v1), &attrs, sizeof(attrs)) == 0,
	      "reported QP size accepted by init_qp");
}

/*
 * Attributes shorter than this build's struct, as passed by a caller compiled
 * against an older header. Fields beyond the caller's inlen must not be read:
 * major 0 treats them as zero, major 1 requires them and rejects.
 */
static void test_short_attrs(struct efa_cuda_host_context *v0,
			     struct efa_cuda_host_context *v1)
{
	const uint32_t old_inlen =
		offsetof(struct efa_cuda_qp_attrs, rq_entry_size) + sizeof(uint32_t);
	uint8_t blob[sizeof(struct efa_cuda_qp_attrs)];
	struct efa_cuda_qp_attrs *attrs = (struct efa_cuda_qp_attrs *)blob;
	struct efa_cuda_qp_v1 qp1;
	struct efa_cuda_qp_v0 qp0;

	/* poison the tail so a stray read past inlen is caught, not masked */
	memset(blob, 0xFF, sizeof(blob));
	memset(attrs, 0, old_inlen);
	attrs->sq_buffer = sq_buf;
	attrs->rq_buffer = rq_buf;
	attrs->sq_doorbell = &sq_db;
	attrs->rq_doorbell = &rq_db;
	attrs->sq_num_entries = 128;
	attrs->rq_num_entries = 32;
	attrs->sq_max_batch = 8;
	attrs->sq_entry_size = 64;
	attrs->rq_entry_size = 16;

	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), attrs, old_inlen) == 0,
	      "major 0 accepts old-header attrs");
	CHECK(qp0.sq.max_inline_data == 0 && qp0.sq.max_rdma_sges == 0,
	      "fields beyond inlen read as zero, not as the poison");
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), attrs, old_inlen) == -EOPNOTSUPP,
	      "major 1 rejects attrs too short to carry capabilities");
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), attrs, 16) == -EINVAL,
	      "attrs shorter than the baseline rejected");
}

static void test_get_version(struct efa_cuda_host_context *, struct efa_cuda_host_context *)
{
	int major, minor, subminor;

	CHECK(efa_cuda_get_version(&major, &minor, &subminor) == 0, "get_version");
	CHECK(major == EFA_CUDA_DP_VERSION_MAJOR && minor == EFA_CUDA_DP_VERSION_MINOR &&
		      subminor == EFA_CUDA_DP_VERSION_SUBMINOR,
	      "get_version reports the build's version");
	CHECK(efa_cuda_get_version(nullptr, &minor, &subminor) == -EINVAL,
	      "get_version rejects NULL");
}

int main()
{
	struct efa_cuda_host_context *v0 = efa_cuda_host_context_create(0, 0, 0);
	struct efa_cuda_host_context *v1 = efa_cuda_host_context_create(1, 0, 0);
	const struct {
		const char *name;
		void (*fn)(struct efa_cuda_host_context *, struct efa_cuda_host_context *);
	} tests[] = {
		{ "context_lifecycle", test_context_lifecycle },
		{ "init_cq", test_init_cq },
		{ "init_qp_v0", test_init_qp_v0 },
		{ "init_qp_v1", test_init_qp_v1 },
		{ "storage_size_mismatch", test_storage_size_mismatch },
		{ "attrs_forward_compat", test_attrs_forward_compat },
		{ "sq_limits", test_sq_limits },
		{ "size_queries", test_size_queries },
		{ "short_attrs", test_short_attrs },
		{ "get_version", test_get_version },
	};

	if (!v0 || !v1) {
		printf("FATAL: context creation failed\n");
		return 1;
	}

	for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
		int fail_before = fail;

		tests[i].fn(v0, v1);
		printf("%-24s %s\n", tests[i].name, fail == fail_before ? "PASS" : "FAIL");
	}

	efa_cuda_host_context_destroy(v0);
	efa_cuda_host_context_destroy(v1);

	printf("\n%s: %d checks passed, %d failed\n", fail ? "FAIL" : "PASS", pass, fail);
	return fail != 0;
}
