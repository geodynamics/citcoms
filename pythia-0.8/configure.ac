#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# $Id: configure.ac,v 1.3 2005/09/28 21:10:01 leif Exp $

AC_PREREQ(2.59)
AC_INIT([Pythia], [0.8], [pyre@cacr.caltech.edu])
AC_CONFIG_AUX_DIR([./aux-config])
AC_CONFIG_SRCDIR([packages/pyre/pyre/applications/Application.py])
AC_CONFIG_HEADER([portinfo])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])

# 'configure' options
PYTHIA_ARG_WITH_MPI
PYTHIA_ARG_WITH_ACIS
AC_ARG_VAR(PYTHON, [Python interpreter])

# Checks for programs.
AM_PATH_PYTHON([2.3])
CIT_PYTHON_SITE
CIT_PYTHON_SYSCONFIG
CIT_PROG_MPICXX
AC_LANG(C++)
AC_PROG_LIBTOOL
CIT_PROG_PYCONFIG

# Checks for libraries.
if test "$want_mpi" = yes; then
    CIT_CHECK_LIB_MPI
fi

# Checks for header files.
if test "$want_mpi" = yes; then
    CIT_HEADER_MPI
    AC_DEFINE(WITH_MPI, 1, [Define if you have an MPI library.])
    AC_DEFINE(PARALLEL, 1, [Define if you have an MPI library.])
    AC_DEFINE(WITH_PARALLEL_IO, 1, [Define if you have an MPI library.])
fi

# Checks for typedefs, structures, and compiler characteristics.
PYTHIA_CXX_FUNCTION
PYTHIA_ACIS_MESHER

# Checks for library functions.

AC_CONFIG_FILES([Makefile
                 examples/Makefile
                 packages/Makefile
                 packages/acis/Makefile
                 packages/acis/tests/Makefile
                 packages/blade/Makefile
                 packages/blade/etc/Makefile
                 packages/elc/Makefile
                 packages/elc/tests/Makefile
                 packages/journal/Makefile
                 packages/journal/journalmodule/Makefile
                 packages/journal/libjournal/Makefile
                 packages/journal/tests/Makefile
                 packages/merlin/Makefile
                 packages/merlin/etc/Makefile
                 packages/mpi/Makefile
                 packages/mpi/mpipython/Makefile
                 packages/mpi/tests/Makefile
                 packages/opal/Makefile
                 packages/opal/tests/Makefile
                 packages/pulse/Makefile
                 packages/pyre/Makefile
                 packages/pyre/etc/Makefile
                 packages/pyre/libpyre/Makefile
                 packages/pyre/tests/Makefile
                 packages/rigid/Makefile
                 tests/Makefile
                 tests/jtest/Makefile])
AC_OUTPUT

dnl end of configure.ac
