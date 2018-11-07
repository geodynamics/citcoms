# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for Proj4.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_PROJ4_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_PROJ4_HEADER], [
  AC_LANG(C)
  AC_CHECK_HEADER([proj_api.h], [], [
    AC_MSG_ERROR([Proj4 header not found; try CPPFLAGS="-I<Proj4 include dir>"])
  ])dnl
])dnl CIT_PROJ4_HEADER


# ----------------------------------------------------------------------
# CIT_PROJ4_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_PROJ4_LIB], [
  AC_LANG(C)
  AC_CHECK_LIB(proj, pj_init_plus, [],[
    AC_MSG_ERROR([Proj4 library not found; try LDFLAGS="-L<Proj4 lib dir>"])
  ])dnl
])dnl CIT_PROJ4_LIB


dnl end of file
