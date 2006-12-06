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

if test "$want_pyre" = yes; then
    # Check for Python.
    AM_PATH_PYTHON([2.3])
    CIT_PYTHON_SYSCONFIG

    # Checks for Python modules and packages.

    builddir=`pwd`
    save_PYTHONPATH="$PYTHONPATH"
    PYTHONPATH="$builddir/python:$PYTHONPATH"; export PYTHONPATH
    cd $srcdir

    AC_MSG_NOTICE([downloading missing Python dependencies])
    AS_IF([AC_TRY_COMMAND([$PYTHON setup.py install_deps -zmaxd $builddir/deps >&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD])],
          [],
          [AC_MSG_FAILURE([cannot download missing Python dependencies])])

    AC_MSG_NOTICE([building Python dependencies])
    AS_IF([AC_TRY_COMMAND([$PYTHON setup.py develop -H None -f $builddir/deps -x -d $builddir/python >&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD])],
          [],
          [AC_MSG_FAILURE([building Python dependencies])])

    AC_MSG_CHECKING([for egg-related flags])
    AS_IF([AC_TRY_COMMAND([$PYTHON setup.py egg_flags >&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD])],
          [AC_MSG_RESULT(ok)
           . egg-flags.sh
           rm -f egg-flags.sh
          ],
          [AC_MSG_RESULT(failed)
          AC_MSG_FAILURE([cannot scan Python eggs for flags])])

    cd $builddir
    PYTHONPATH="$save_PYTHONPATH"
    PYTHONPATH="${pythondir}:${pyexecdir}${save_PYTHONPATH:+:${save_PYTHONPATH}}"

    AC_SUBST(PYTHONPATH)
    AC_SUBST(PYTHON_EGG_CFLAGS)
    AC_SUBST(PYTHON_EGG_CPPFLAGS)
    AC_SUBST(PYTHON_EGG_LDFLAGS)

fi

# Checks for programs.
AC_PROG_CC([mpicc hcc mpcc mpcc_r mpxlc cmpicc gcc cc cl icc ecc pgcc xlc xlc_r])
if test "$want_pyre" = yes; then
    CIT_PROG_PYCONFIG
    AC_SUBST([pkgsysconfdir], [\${sysconfdir}/$PACKAGE])
fi
# We don't need Fortran.
AC_PROVIDE([AC_PROG_F77])
AC_PROG_LIBTOOL

# Checks for libraries.
AC_SEARCH_LIBS([MPI_Init], [mpi mpich], [], [AC_MSG_ERROR([MPI library not found])])
AC_SEARCH_LIBS([sqrt], [m])
CIT_CHECK_LIB_HDF5
CIT_CHECK_LIB_HDF5_PARALLEL

# Checks for header files.
AC_CHECK_HEADER([mpi.h], [], [AC_MSG_ERROR([header 'mpi.h' not found])])
AC_CHECK_HEADERS([malloc.h])
CIT_CHECK_HEADER_HDF5

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AM_CONDITIONAL([COND_HDF5], [test "$want_hdf5" != no])

AC_CONFIG_FILES([Makefile
                 bin/Makefile
                 CitcomS/Makefile
                 etc/Makefile
                 examples/Makefile
                 lib/Makefile
                 module/Makefile
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
echo -e "\t LDFLAGS: " $LDFLAGS
echo -e "\t with-pyre: " $want_pyre
echo -e "\t with-hdf5: " $want_hdf5
echo

dnl end of configure.ac
