dnl                                               -*- Autoconf -*-
dnl Process this file with autoconf to produce a configure script.
dnl
dnl<LicenseText>
dnl
dnl CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
dnl Copyright (C) 2002-2005, California Institute of Technology.
dnl
dnl This program is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 2 of the License, or
dnl (at your option) any later version.
dnl
dnl This program is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with this program; if not, write to the Free Software
dnl Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
dnl
dnl</LicenseText>

# $Id$

AC_PREREQ(2.59)
AC_INIT([CitcomS], [2.0.1], [cig-mc@geodynamics.org], [CitcomS])
AC_CONFIG_AUX_DIR([./aux-config])
AC_CONFIG_SRCDIR([drivers/Citcom.c])
AC_CONFIG_HEADER([portinfo])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])

# 'configure' options
AC_ARG_VAR(PYTHON, [Python interpreter])
AC_ARG_VAR(PYTHONPATH, [Python module search path])
AC_ARG_VAR(MPIRUN, [command to run MPI programs])
AC_ARG_VAR(BSUB, [command to submit LSF batch jobs])
AC_ARG_VAR(QSUB, [command to submit PBS batch jobs])
AC_ARG_VAR(GLOBUSRUN, [command to submit Globus batch jobs])
AC_ARG_VAR(BATCHTYPE, [default batch system (lsf, pbs, or globus)])
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
    AC_PATH_PROG([MPIRUN], [mpirun])
    if test -z "$MPIRUN"; then
        AC_MSG_ERROR([program 'mpirun' not found])
    fi
    AC_PATH_PROG([BSUB], [bsub])
    AC_PATH_PROG([QSUB], [qsub])
    AC_PATH_PROG([GLOBUSRUN], [globusrun])
    if test -z "$BATCHTYPE"; then
        if test -n "$BSUB"; then
            LAUNCHER="lsf"
        elif test -n "$QSUB"; then
            LAUNCHER="pbs"
        elif test -n "$GLOBUSRUN"; then
            LAUNCHER="globus"
        else
            if $MPIRUN --help 2>&1 | grep machinefile >/dev/null; then
                LAUNCHER="mpich"
            else
                LAUNCHER="lam-mpi"
            fi
        fi
        test -z "$BSUB" && BSUB="bsub"
        test -z "$QSUB" && QSUB="qsub"
        test -z "$GLOBUSRUN" && GLOBUSRUN="globusrun"
    else
        case "$BATCHTYPE" in
            lsf | pbs | globus) LAUNCHER=$BATCHTYPE ;;
            *) AC_MSG_ERROR([BATCHTYPE=$BATCHTYPE is invalid]) ;;
        esac
    fi
    AC_SUBST([LAUNCHER])
fi
CIT_PROG_MPICC
CIT_PROG_MPICXX
if test "$want_pyre" = yes; then
    AC_CHECK_PROG(check_CXX, [$CXX], [$CXX], [ ])
    if test "$check_CXX" = " "; then
        AC_MSG_ERROR([no C++ compiler found])
    fi
    CIT_PROG_PYCONFIG
    AC_SUBST([pkgsysconfdir], [\${sysconfdir}/$PACKAGE])
fi
AC_PROG_LIBTOOL

# Checks for Python modules and packages.
if test "$want_pyre" = yes; then
    PYTHONPATH="${pythondir}${PYTHONPATH:+:${PYTHONPATH}}"; export PYTHONPATH
    CIT_PATH_PYTHIA([0.8], [mpi])
    AC_SUBST([PYTHONPATH])
fi

# Checks for libraries.
if test "$want_pyre" = yes; then
    AC_LANG_PUSH(C++)
    CIT_CHECK_LIB_MPI
    AC_LANG_POP(C++)
else
    CIT_CHECK_LIB_MPI
fi

# Checks for header files.
if test "$want_pyre" = yes; then
    AC_LANG_PUSH(C++)
    CIT_HEADER_MPI
    AC_LANG_POP(C++)
else
    CIT_HEADER_MPI
fi
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
                 module/Full/Makefile
                 module/Regional/Makefile
                 pyre/Makefile
                 tests/Makefile
                 visual/Makefile])

if test "$want_pyre" = yes; then
    AC_CONFIG_SUBDIRS([pythia-0.8])
fi

AC_OUTPUT

dnl end of configure.ac
