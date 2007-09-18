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
AC_INIT([CitcomS], [2.2.1], [cig-mc@geodynamics.org], [CitcomS])
AC_CONFIG_AUX_DIR([./aux-config])
AC_CONFIG_SRCDIR([bin/Citcom.c])
AC_CONFIG_HEADER([config.h])
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
AC_ARG_ENABLE([embedding],
    [AC_HELP_STRING([--enable-embedding],
        [embed Python with CitcomS in a single executable @<:@default=yes@:>@])],
    [want_embedding="$enableval"],
    [want_embedding=yes])
AM_CONDITIONAL([COND_EMBEDDING], [test "$want_embedding" = yes])
CIT_ARG_HDF5([auto])
AC_ARG_WITH([exchanger],
    [AC_HELP_STRING([--with-exchanger],
        [use Exchanger @<:@default=auto@:>@])],
    [want_exchanger="$withval"],
    [want_exchanger=auto])

if test "$want_pyre" = yes; then
    # Check for Python.
    AM_PATH_PYTHON([2.3])
    CIT_PYTHON_SYSCONFIG

    # Check for Python modules and packages.
    export want_exchanger
    CIT_PYTHON_EGG_SETUP
else
    if test "$want_exchanger" = yes; then
        AC_MSG_ERROR([Exchanger requires Pyre])
    fi
    want_exchanger=no
fi

# Checks for programs.
AC_PROG_CC([mpicc hcc mpcc mpcc_r mpxlc cmpicc gcc cc cl icc ecc pgcc xlc xlc_r])
AC_PROG_CXX([mpicxx mpic++ mpiCC hcp mpCC mpxlC mpxlC_r cmpic++ g++ c++ gpp aCC CC cxx cc++ cl FCC KCC RCC xlc++_r xlC_r xlC icpc ecpc pgCC])
if test "$want_pyre" = yes; then
    CIT_PROG_PYCONFIG
    AC_SUBST([pkgsysconfdir], [\${sysconfdir}/$PACKAGE])
    # Compile C as C++.
    dnl CC="$CXX"
    dnl CFLAGS="$CXXFLAGS"
fi
# We don't need Fortran.
AC_PROVIDE([AC_PROG_F77])
AC_PROG_LIBTOOL

# Checks for libraries.
AC_SEARCH_LIBS([MPI_Init], [mpi mpich], [], [AC_MSG_ERROR([MPI library not found])])
AC_SEARCH_LIBS([sqrt], [m])

CIT_CHECK_LIB_HDF5
CIT_CHECK_LIB_HDF5_PARALLEL

if test "$want_exchanger" != no; then
    save_CC="$CC"
    save_CFLAGS="$CFLAGS"
    save_LIBS="$LIBS"
    CC="$CXX"
    CFLAGS="$CXXFLAGS"
    LIBS="$PYTHON_EGG_LDFLAGS $PYTHON_BLDLIBRARY $PYTHON_LIBS $PYTHON_MODLIBS $PYTHON_SYSLIBS $LIBS"
    AC_CHECK_FUNC([PyExchanger_exchangeBoundedBox], [], [
        if test "$want_exchanger" = auto; then
            want_exchanger=no
            AC_MSG_WARN([Exchanger library not found; disabling Exchanger support])
        else
            AC_MSG_ERROR([Exchanger library not found])
        fi
    ])
    CC="$save_CC"
    CFLAGS="$save_CFLAGS"
    LIBS="$save_LIBS"
fi

# Checks for header files.
AC_CHECK_HEADER([mpi.h], [], [AC_MSG_ERROR([header 'mpi.h' not found])])
AC_CHECK_HEADERS([malloc.h])
CIT_CHECK_HEADER_HDF5

AC_LANG_PUSH(C++)
if test "$want_exchanger" != no; then
    CIT_HEADER_MPI
    save_CPPFLAGS="$CPPFLAGS"
    CPPFLAGS="$PYTHON_EGG_CPPFLAGS $CPPFLAGS"
    AC_CHECK_HEADERS([Exchanger/DIM.h], [], [
        if test "$want_exchanger" = auto; then
            want_exchanger=no
            AC_MSG_WARN([header 'Exchanger/DIM.h' not found; disabling Exchanger support])
        else
            AC_MSG_ERROR([header 'Exchanger/DIM.h' not found])
        fi
    ])
    CPPFLAGS="$save_CPPFLAGS"
fi
AC_LANG_POP()

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

if test "$want_hdf5" != no; then
    want_hdf5=yes
fi
if test "$want_exchanger" != no; then
    want_exchanger=yes
fi
AM_CONDITIONAL([COND_HDF5], [test "$want_hdf5" = yes])
AM_CONDITIONAL([COND_EXCHANGER], [test "$want_exchanger" = yes])

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
echo -e "\t CC: " $CC
echo -e "\t CFLAGS: " $CFLAGS
echo -e "\t CPPFLAGS: " $CPPFLAGS
echo -e "\t CXX: " $CXX
echo -e "\t CXXFLAGS: " $CXXFLAGS
echo -e "\t LDFLAGS: " $LDFLAGS
echo -e "\t with-pyre: " $want_pyre
echo -e "\t with-exchanger: " $want_exchanger
echo -e "\t with-hdf5: " $want_hdf5
echo

dnl end of configure.ac
