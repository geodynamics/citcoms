# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for spatialdata.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_SPATIALDATA_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_SPATIALDATA_HEADER], [
  AC_LANG(C++)
  AC_CHECK_HEADER([spatialdata/spatialdb/SpatialDB.hh], [], [
    AC_MSG_ERROR([SpatialDB header not found; try CPPFLAGS="-I<Spatialdata include dir>"])
  ])dnl
])dnl CIT_SPATIALDATA_HEADER


# ----------------------------------------------------------------------
# CIT_SPATIALDATA_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_SPATIALDATA_LIB], [
  AC_LANG(C++)
  AC_REQUIRE_CPP
  AC_MSG_CHECKING([for SimpleDB in -lspatialdata])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <spatialdata/spatialdb/SpatialDB.hh>]
                      [#include <spatialdata/spatialdb/SimpleDB.hh>]],
                     [[spatialdata::spatialdb::SimpleDB db;]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([Spatialdata library not found; try LDFLAGS="-L<Spatialdata lib dir>"])
  ])dnl
])dnl CIT_SPATIALDATA_LIB


dnl end of file
