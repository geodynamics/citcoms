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
AC_INIT([CitcomS], [2.1.0], [cig-mc@geodynamics.org], [CitcomS])
AC_CONFIG_AUX_DIR([./aux-config])
AC_CONFIG_SRCDIR([bin/Citcom.c])
AC_CONFIG_HEADER([portinfo])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])

# 'configure' options
AC_ARG_VAR(PYTHON, [Python interpreter])
AC_ARG_VAR(PYTHONPATH, [Python module search path])
AC_ARG_VAR(PHDF5_HOME, [home path to parallel HDF5 library])
AC_ARG_WITH([pyre],
    [AC_HELP_STRING([--with-pyre],
        [build Pyre modules @<:@default=yes@:>@])],
    [want_pyre="$withval"],
    [want_pyre=yes])
AM_CONDITIONAL([COND_PYRE], [test "$want_pyre" = yes])
AC_ARG_WITH([exchanger],
    [AC_HELP_STRING([--with-exchanger],
        [use Exchanger @<:@default=auto@:>@])],
    [want_exchanger="$withval"],
    [want_exchanger=auto])
AC_ARG_WITH([hdf5],
    [AC_HELP_STRING([--with-hdf5],
        [enable HDF5 output @<:@default=auto@:>@])],
    [want_hdf5="$withval"],
    [want_hdf5=auto])
AC_ARG_ENABLE([embedding],
    [AC_HELP_STRING([--enable-embedding],
        [embed Python with CitcomS in a single executable @<:@default=yes@:>@])],
    [want_embedding="$enableval"],
    [want_embedding=yes])
AM_CONDITIONAL([COND_EMBEDDING], [test "$want_embedding" = yes])

if test "$want_pyre" = yes; then
    # Check for Python.
    AM_PATH_PYTHON([2.3])
    CIT_PYTHON_SYSCONFIG

    # Checks for Python modules and packages.
    CIT_CHECK_PYTHON_EGG([pythia >= 0.8.1.0b2, < 0.8-2.0a])
    if test "$want_exchanger" != no; then
        CIT_CHECK_PYTHON_EGG([Exchanger >= 1.0, < 2.0a],
                             [ # found
                                 if test "$want_exchanger" = auto; then
                                     want_exchanger=yes
                                 fi
                             ], [ # not found
                                 if test "$want_exchanger" = auto; then
                                     want_exchanger=no
                                     AC_MSG_WARN(['Exchanger' package not found; disabling Exchanger support])
                                 else
                                     AC_MSG_ERROR(['Exchanger' package not found])
                                 fi
			     ])
    fi
    CIT_PYTHON_EGG_FLAGS
else
    want_exchanger=no
fi
AM_CONDITIONAL([COND_EXCHANGER], [test "$want_exchanger" = yes])

# Checks for programs.
CIT_PROG_MPICC
CIT_PROG_MPICXX
CC=$MPICC
CXX=$MPICXX
if test "$want_pyre" = yes; then
    CIT_PROG_PYCONFIG
    AC_SUBST([pkgsysconfdir], [\${sysconfdir}/$PACKAGE])
fi
# We don't need Fortran.
AC_PROVIDE([AC_PROG_F77])
AC_PROG_LIBTOOL
AC_LANG(C)

# Checks for libraries.
if test "$want_exchanger" = yes; then
    AC_LANG_PUSH(C++)
    CIT_CHECK_LIB_MPI
    AC_LANG_POP(C++)
else
    CIT_CHECK_LIB_MPI
fi

# Checks for header files.
if test "$want_exchanger" = yes; then
    AC_LANG_PUSH(C++)
    CIT_HEADER_MPI
    AC_LANG_POP(C++)
else
    CIT_HEADER_MPI
fi
if test "$want_hdf5" != no; then
    if test "$PHDF5_HOME"; then
        CPPFLAGS="-I$PHDF5_HOME/include $CPPFLAGS"
    fi
    AC_CHECK_HEADERS([hdf5.h], [AC_DEFINE([HAVE_HDF5_H])], [HDF5NOTFOUND=1])
    if test "$HDF5NOTFOUND"; then
        if test "$want_hdf5" = auto; then
            want_hdf5=no
            AC_MSG_WARN([Cannot find hdf5.h, disabling HDF5 support])
        else
            AC_MSG_ERROR([HDF5 header file "hdf5.h" not found!])
        fi
    fi
fi
AC_CHECK_HEADERS([malloc.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_ARG_VAR(LIBM, [math library])
if test -z "$LIBM"; then
    AC_CHECK_FUNC(sqrt, , [
        AC_CHECK_LIB(m, sqrt, [
            LIBM="-lm"
        ], [
            AC_MSG_ERROR([math library not found; try setting LIBM])
        ])
    ])
fi
AC_SUBST(LIBM)

if test "$want_hdf5" != no; then
    if test "$PHDF5_HOME"; then
        LDFLAGS="-L$PHDF5_HOME/lib $LDFLAGS"
    fi
    # Checks for basic HDF5 function
    AC_CHECK_FUNC(H5Fopen, , [
        AC_CHECK_LIB(hdf5, H5Fopen, [LIBHDF5="-lhdf5"], [HDF5NOTFOUND=1])
    ])
    if test "$HDF5NOTFOUND"; then
        if test "$want_hdf5" = auto; then
            want_hdf5=no
            AC_MSG_WARN([Cannot find HDF5 library, disabling HDF5 support])
        else
            AC_MSG_ERROR([HDF5 library not found; try setting PHDF5_HOME])
        fi
    fi
fi
if test "$want_hdf5" != no; then
    # Checks for HDF5 parallel-IO function
    AC_CHECK_LIB(hdf5, H5Pset_dxpl_mpio, [have_phdf5="yes"], [HDF5NOTFOUND=1])
    if test "$HDF5NOTFOUND"; then
        if test "$want_hdf5" = auto; then
            want_hdf5=no
            AC_MSG_WARN([Cannot find parallel HDF5 library, disabling HDF5 support])
        else
            AC_MSG_ERROR([Need Parallel HDF5; try compiling HDF5 with --enable-parallel])
        fi
    fi
    if test "$want_hdf5" = auto; then
	want_hdf5=yes
    fi
fi
AM_CONDITIONAL([COND_HDF5], [test "$want_hdf5" = yes])
AC_SUBST(LIBHDF5)

AC_CONFIG_FILES([Makefile
                 bin/Makefile
                 CitcomS/Makefile
                 etc/Makefile
                 examples/Makefile
                 lib/Makefile
                 module/Makefile
                 module/Exchanger/Makefile
                 tests/Makefile
                 visual/Makefile
                 visual/OpenDXMacro/Makefile])

AC_OUTPUT

# report configuration summary
echo -e "\n\n"
echo "================ Configuration Summary ================"
echo -e "\t PYTHON: " $PYTHON
echo -e "\t PYTHONPATH: " $PYTHONPATH
echo -e "\t MPICC: " $MPICC
if test "$want_exchanger" = yes; then
   echo -e "\t MPICXX: " $MPICXX
fi
echo -e "\t CFLAGS: " $CFLAGS
echo -e "\t CPPFLAGS: " $CPPFLAGS
echo -e "\t LDFLAGS: " $LDFLAGS
if test "$want_exchanger" = yes; then
    echo -e "\t CXXFLAGS: " $CXXFLAGS
fi
echo -e "\t with-pyre: " $want_pyre
echo -e "\t with-exchanger: " $want_exchanger
echo -e "\t with-hdf5: " $want_hdf5
echo

dnl end of configure.ac
