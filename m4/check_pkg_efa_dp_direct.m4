# -*- autoconf -*-
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_EFA_DP_DIRECT], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([efa-dp-direct],
     [AS_HELP_STRING([--with-efa-dp-direct=PATH],
                     [Path to efa-dp-direct install. The PATH is expected to contain an include/ subdirectory with efa_cuda_dp.h and a lib/ (or lib64/) subdirectory with libefacudadp.so. The package is needed for the GIN GDAKI data path; if not provided, the GDAKI context APIs will compile out.])])

  AS_IF([test -z "${with_efa_dp_direct}" -o "${with_efa_dp_direct}" = "yes"],
        [],
        [test "${with_efa_dp_direct}" = "no"],
        [check_pkg_found=no],
        [dnl efa-dp-direct's Makefile produces build/include/ and build/libefacudadp.so
         dnl in the same directory (no lib/ subdir). Accept either the build tree
         dnl layout (PATH/include + PATH/libefacudadp.so) or a standard install
         dnl layout (PATH/include + PATH/lib[64]/libefacudadp.so).
         AS_IF([test -f ${with_efa_dp_direct}/libefacudadp.so],
               [check_pkg_libdir=""
                efa_dp_direct_libpath="${with_efa_dp_direct}"],
               [test -d ${with_efa_dp_direct}/lib64],
               [check_pkg_libdir="/lib64"
                efa_dp_direct_libpath="${with_efa_dp_direct}/lib64"],
               [check_pkg_libdir="/lib"
                efa_dp_direct_libpath="${with_efa_dp_direct}/lib"])
         CPPFLAGS="-isystem ${with_efa_dp_direct}/include ${CPPFLAGS}"
         LDFLAGS="-L${efa_dp_direct_libpath} ${LDFLAGS}"])

  dnl efa_cuda_dp.h includes <cuda_runtime.h>, so make sure CUDA's include
  dnl path is visible during header and link checks.
  AS_IF([test "${check_pkg_found}" = "yes"],
        [CPPFLAGS="${CUDA_CPPFLAGS} ${CPPFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([efa_cuda_dp.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS([efa_cuda_get_version], [efacudadp], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         LDFLAGS="${check_pkg_LDFLAGS_save}"
         LIBS="${check_pkg_LIBS_save}"])

  AC_DEFINE_UNQUOTED([HAVE_EFA_DP_DIRECT], [${check_pkg_define}],
                     [Defined to 1 if efa-dp-direct is available for the GIN GDAKI data path])
  AM_CONDITIONAL([HAVE_EFA_DP_DIRECT], [test ${check_pkg_define} = 1])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
  AS_UNSET([efa_dp_direct_libpath])
])
