# -*- Autoconf -*-


## --------------------------- ##
## Autoconf macros for CUDA. ##
## --------------------------- ##

# ----------------------------------------------------------------------
# CIT_CUDA_INCDIR
# ----------------------------------------------------------------------
# Determine the directory containing <cuda_runtime.h>
AC_DEFUN([CIT_CUDA_INCDIR], [
  AC_REQUIRE_CPP
  AC_LANG(C++)
  AC_CHECK_HEADER([cuda_runtime.h], [], [
    AC_MSG_ERROR([CUDA runtime header not found; try CPPFLAGS="-I<CUDA include dir>"])
  ])dnl
])dnl CIT_CUDA_INCDIR


# ----------------------------------------------------------------------
# CIT_CUDA_LIB
# ----------------------------------------------------------------------
# Checking for the CUDA library.
AC_DEFUN([CIT_CUDA_LIB], [
  AC_REQUIRE_CPP
  AC_LANG(C++)
  AC_MSG_CHECKING([for cudaMalloc in -lcuda])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <cuda_runtime.h>]],
                     [[void* ptr = 0;]]
  	             [[cudaMalloc(&ptr, 1);]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([cuda library not found.])
  ])dnl
])dnl CIT_CUDA_LIB

# ----------------------------------------------------------------------
# CIT_CUDA_COMPILER
# ----------------------------------------------------------------------
# Checking for the CUDA compiler.
AC_DEFUN([CIT_CUDA_COMPILER], [
  AC_PATH_PROG(NVCC, nvcc)
  if test -z "$NVCC" ; then
    AC_MSG_FAILURE([cannot find 'nvcc' program.])
    NVCC=`echo "Error: nvcc is not installed." ; false`
  fi
])dnl CIT_CUDA_COMPILER



dnl end of file
