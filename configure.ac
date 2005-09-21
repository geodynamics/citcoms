#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# $Id$

AC_PREREQ(2.59)
AC_INIT([CitcomS], [2.1.0], [cig-mc@geodynamics.org], [CitcomS])
AC_CONFIG_AUX_DIR([./aux-config])
AC_CONFIG_SRCDIR([drivers/Citcom.c])
AC_CONFIG_HEADER([portinfo])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])

# 'configure' options
AC_ARG_VAR(PYTHON, [Python interpreter])
AC_ARG_VAR(PYTHONPATH, [Python module search path])
AC_ARG_WITH([pyre],
    [AC_HELP_STRING([--with-pyre],
        [build Pyre modules @<:@default=yes@:>@])],
    [want_pyre="$withval"],
    [want_pyre=yes])
AM_CONDITIONAL([COND_PYRE], [test "$want_pyre" = yes])

# Checks for programs.
if test "$want_pyre" = yes; then
    AM_PATH_PYTHON([2.3])
    CIT_PYTHON_INCDIR
fi
CIT_PROG_MPICC
CIT_PROG_MPICXX
AC_PROG_LIBTOOL

# Checks for Python modules and packages.
if test "$want_pyre" = yes; then
    CIT_PATH_PYTHIA([0.8], [mpi])
    CIT_PATH_EXCHANGER([1.0])
fi

# Checks for libraries.
CIT_CHECK_LIB_MPI

# Checks for header files.
CIT_HEADER_MPI
AC_CHECK_HEADERS([malloc.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_ARG_VAR(LIBM, [math library])
if test -z "$LIBM"; then
    AC_CHECK_FUNC(sqrt, , [
        AC_CHECK_LIB(m, sqrt, [
            LIBM="-lm"
        ])
    ])
fi
AC_SUBST(LIBM)

AC_CONFIG_FILES([Makefile
                 drivers/Makefile
                 etc/Makefile
                 examples/Makefile
                 lib/Makefile
                 lib/Common/Makefile
                 lib/Full/Makefile
                 lib/Regional/Makefile
                 module/Makefile
                 module/Exchanger/Makefile
                 module/Full/Makefile
                 module/Regional/Makefile
                 pyre/Makefile
                 tests/Makefile
                 visual/Makefile])

if test "$want_pyre" = yes; then
    AC_CONFIG_SUBDIRS([pythia-0.8 Exchanger])
fi

AC_OUTPUT

dnl end of configure.ac
