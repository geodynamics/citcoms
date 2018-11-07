# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for ADIOS.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_ADIOS_CONFIG
# ----------------------------------------------------------------------
AC_DEFUN([CIT_ADIOS_CONFIG], [
  dnl ADIOS comes with a program that *should* tell us how to link with it.
  AC_ARG_VAR([ADIOS_CONFIG], [Path to adios_config program that indicates how to compile with it.])
  AC_PATH_PROG([ADIOS_CONFIG], [adios_config])

  if test "x$ADIOS_CONFIG" = "x"; then
    AC_MSG_ERROR([adios_config program not found; try setting ADIOS_CONFIG to point to it])
  fi

  AC_LANG_PUSH([Fortran])
  FC_save="$FC"
  FCFLAGS_save="$FCFLAGS"
  LIBS_save="$LIBS"
  FC="$MPIFC" dnl Must use mpi compiler.

  dnl First check for directory with ADIOS modules
  AC_MSG_CHECKING([for ADIOS modules])
  ADIOS_FCFLAGS=`$ADIOS_CONFIG -c -f`
  FCFLAGS="$ADIOS_FCFLAGS $FCFLAGS"
  AC_COMPILE_IFELSE([
    AC_LANG_PROGRAM([], [[
    use adios_read_mod
    use adios_write_mod
    ]])
  ], [
    AC_MSG_RESULT(yes)
  ], [
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([ADIOS modules not found; is ADIOS built with Fortran support for this compiler?])
  ])

  dnl Now check for libraries that must be linked.
  AC_MSG_CHECKING([for ADIOS libraries])
  FCFLAGS="$ADIOS_FCFLAGS $FCFLAGS_save"
  ADIOS_LIBS=`$ADIOS_CONFIG -l -f`
  LIBS="$ADIOS_LIBS $LIBS"
  AC_LINK_IFELSE([
    AC_LANG_PROGRAM([],
                    [[use adios_write_mod; call adios_init_noxml(MPI_COMM_WORLD, ierr)]])
  ], [
    AC_MSG_RESULT(yes)
  ], [
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([ADIOS libraries not found.])
  ])

  FC="$FC_save"
  FCFLAGS="$FCFLAGS_save"
  LIBS="$LIBS_save"
  AC_LANG_POP([Fortran])

  AC_SUBST([ADIOS_FCFLAGS])
  AC_SUBST([ADIOS_LIBS])
])dnl CIT_ADIOS_CONFIG

dnl end of file
