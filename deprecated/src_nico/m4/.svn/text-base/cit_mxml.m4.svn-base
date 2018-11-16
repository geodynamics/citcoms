# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for netcdf.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_MXML_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_MXML_HEADER], [
  AC_LANG(C)
  AC_CHECK_HEADER([mxml.h], [], [
    AC_MSG_ERROR([mxml C header not found; try CPPFLAGS="-I<mxml include dir>"])
  ])dnl
])dnl CIT_MXML_HEADER


# ----------------------------------------------------------------------
# CIT_MXML_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_MXML_LIB], [
  AC_LANG(C)
  AC_REQUIRE_CPP
  AC_MSG_CHECKING([for mxmlNewXML in -lmxml])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <mxml.h>]],
	             [[mxml_node_t *xml = mxmlNewXML("1.0");]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([mxml library not found; try LDFLAGS="-L<mxml lib dir>"])
    ])dnl
  ]))
])dnl CIT_MXML_LIB


dnl end of file
