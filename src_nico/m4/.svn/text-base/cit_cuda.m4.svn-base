# -*- Autoconf -*-


## ------------------------- ##
## Autoconf macros for CUDA. ##
## ------------------------- ##


# ----------------------------------------------------------------------
# CIT_CUDA_CONFIG
# ----------------------------------------------------------------------
# Determine the directory containing <cuda_runtime.h>
AC_DEFUN([CIT_CUDA_CONFIG], [
  AC_ARG_VAR(CUDA_INC, [Location of CUDA include files])
  AC_ARG_VAR(CUDA_LIB, [Location of CUDA library libcudart])

  dnl Check for compiler
  AC_PATH_PROG(NVCC, nvcc)
  if test -z "$NVCC" ; then
    AC_MSG_ERROR([cannot find 'nvcc' program.])
  fi

  AC_LANG_PUSH([C++])
  AC_REQUIRE_CPP
  CPPFLAGS_save="$CPPFLAGS"
  LDFLAGS_save="$LDFLAGS"
  LIBS_save="$LIBS"

  dnl Check for CUDA headers
  if test "x$CUDA_INC" != "x"; then
    CUDA_CPPFLAGS="-I$CUDA_INC"
    CPPFLAGS="$CUDA_CPPFLAGS $CPPFLAGS"
  fi
  AC_CHECK_HEADER([cuda_runtime.h], [], [
    AC_MSG_ERROR([CUDA runtime header not found; try setting CUDA_INC.])
  ])

  if test "x$CUDA_LIB" != "x"; then
    CUDA_LDFLAGS="-L$CUDA_LIB"
    LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"
  fi
  CUDA_LIBS="-lcudart"
  LIBS="$CUDA_LIBS $LIBS"
  AC_MSG_CHECKING([for cudaMalloc in -lcudart])
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[#include <cuda_runtime.h>]],
                     [[void* ptr = 0;]]
  	             [[cudaMalloc(&ptr, 1);]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([CUDA library not found; try setting CUDA_LIB.])
  ])

  CPPFLAGS="$CPPFLAGS_save"
  LDFLAGS="$LDFLAGS_save"
  LIBS="$LIBS_save"
  AC_LANG_POP([C++])

  AC_SUBST([CUDA_CPPFLAGS])
  AC_SUBST([CUDA_LDFLAGS])
  AC_SUBST([CUDA_LIBS])
])dnl CIT_CUDA_COMPILER


dnl end of file
