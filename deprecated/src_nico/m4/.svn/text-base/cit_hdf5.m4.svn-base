# -*- Autoconf -*-

# ======================================================================
# Autoconf macros for HDF5.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_HDF5_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_HDF5_HEADER], [
  AC_LANG(C++)
  AC_CHECK_HEADER([hdf5.h], [], [
    AC_MSG_ERROR([HDF5 header not found; try CPPFLAGS="-I<hdf5 include dir>"])
  ])dnl
])dnl CIT_HDF5_HEADER


# ----------------------------------------------------------------------
# CIT_NETCDF_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_HDF5_LIB], [
  AC_LANG(C++)
  AC_REQUIRE_CPP
  AC_MSG_CHECKING([for H5Fopen in -lhdf5])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <hdf5.h>]],
	             [[H5Fopen("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT);]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([hdf5 library not found; try LDFLAGS="-L<hdf5 lib dir>"])
    ])dnl
  ]))
])dnl CIT_HDF5_LIB


# ----------------------------------------------------------------------
# CIT_NETCDF_LIB_PARALLEL
# ----------------------------------------------------------------------
AC_DEFUN([CIT_HDF5_LIB_PARALLEL], [
  AC_LANG(C++)
  AC_REQUIRE_CPP
  AC_SEARCH_LIBS([H5Pset_dxpl_mpio], [hdf5], [], [
    AC_MSG_WARN([parallel HDF5 library not found; DO NOT attempt to use HDF5 in parallel OR configure HDF5 with '--enable-parallel'])
  ])
])dnl CIT_HDF5_LIB


dnl end of file
