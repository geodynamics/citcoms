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
AC_CONFIG_SRCDIR([Citcom.c])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE([foreign])

# Checks for programs.
AC_PROG_CC([mpicc hcc mpcc mpcc_r mpxlc cmpicc gcc cc cl icc ecc pgcc xlc xlc_r])
AC_PROG_RANLIB

# Checks for libraries.
AC_SEARCH_LIBS([MPI_Init], [mpi mpich], [], [AC_MSG_ERROR([MPI library not found])])
AC_SEARCH_LIBS([sqrt], [m])

# Checks for header files.
AC_CHECK_HEADER([mpi.h], [], [AC_MSG_ERROR([header 'mpi.h' not found])])
AC_CHECK_HEADERS([malloc.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AC_CONFIG_FILES([Makefile
                 examples/Makefile
                 tests/Makefile
                 visual/Makefile
                 visual/OpenDXMacro/Makefile])

AC_OUTPUT

# Warning!
AC_MSG_WARN([to build the Pyrized version, run 'merlin configure' instead])

dnl end of configure.ac
