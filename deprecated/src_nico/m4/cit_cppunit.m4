# -*- Autoconf -*-


# ======================================================================
# Autoconf macros for cppunit.
# ======================================================================

# ----------------------------------------------------------------------
# CIT_CPPUNIT_HEADER
# ----------------------------------------------------------------------
AC_DEFUN([CIT_CPPUNIT_HEADER], [
  AC_LANG(C++)
  AC_CHECK_HEADER([cppunit/TestRunner.h], [], [
    AC_MSG_ERROR([CppUnit header not found; try CPPFLAGS="-I<CppUnit include dir>"])
  ])dnl
])dnl CIT_CPPUNIT_HEADER


# ----------------------------------------------------------------------
# CIT_CPPUNIT_LIB
# ----------------------------------------------------------------------
AC_DEFUN([CIT_CPPUNIT_LIB], [
  AC_LANG(C++)
  AC_MSG_CHECKING([for CppUnit::TestRunner in -lcppunit])
  AC_REQUIRE_CPP
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <cppunit/TestRunner.h>]],
	             [[CppUnit::TestRunner runner;]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([CppUnit library not found; try LDFLAGS="-L<CppUnit lib dir>"])
  ])dnl
])dnl CIT_CPPUNIT


dnl end of file
