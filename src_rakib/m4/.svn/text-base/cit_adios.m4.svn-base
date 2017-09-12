# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for ADIOS.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_ADIOS_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_ADIOS_HEADER], [
  AC_REQUIRE_CPP
  AC_LANG(C)
  AC_CHECK_HEADER([adios.h], [], [
    AC_MSG_ERROR([adios C header not found; try CPPFLAGS="-I<adios include dir>"])
  ])dnl
])dnl CIT_ADIOS_HEADER


# ----------------------------------------------------------------------
# CIT_ADIOS_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_ADIOS_LIB], [
  AC_LANG(Fortran)
  AC_MSG_CHECKING([for adios_init in -ladiosf])
  dnl Sample ADIOS program must be compiled with mpif90 in order to link
  dnl the proper libraries (eg the ones bundled with openmpi)
  FC_BCK=$FC
  FC=$MPIFC
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[]],
	             [[use adios_write_mod; call adios_init_noxml(MPI_COMM_WORLD, ierr)]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([adiosf library not found; try LDFLAGS="-L<adios lib dir>" 
					     and FCFLAGS="-I<adios inc dir>"])
    ])dnl
  dnl Revert the Fortran compiler to its initial value
  FC=$FC_BCK
  ]))
])dnl CIT_ADIOS_LIB


dnl end of file
