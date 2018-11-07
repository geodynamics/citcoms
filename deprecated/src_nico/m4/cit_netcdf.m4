# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for netcdf.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_NETCDF_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_NETCDF_HEADER], [
  AC_LANG(C++)
  AC_CHECK_HEADER([netcdfcpp.h], [], [
    AC_MSG_ERROR([netcdf C++ header not found; try CPPFLAGS="-I<netcdf include dir>"])
  ])dnl
])dnl CIT_NETCDF_HEADER


# ----------------------------------------------------------------------
# CIT_NETCDF_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_NETCDF_LIB], [
  AC_LANG(C++)
  AC_REQUIRE_CPP
  AC_MSG_CHECKING([for NcFile in -lnetcdfc++])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <netcdfcpp.h>]],
	             [[NcFile ncfile("filename");]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([netcdfc++ library not found; try LDFLAGS="-L<netcdf lib dir>"])
    ])dnl
  ]))
])dnl CIT_NETCDF_LIB


dnl end of file
