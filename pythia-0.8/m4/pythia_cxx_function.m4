# PYTHIA_CXX_FUNCTION
# -------------------
# If the compiler supports __FUNCTION__, define HAVE__FUNC__ to 1.
# If the compiler doesn't support __FUNCTION__ but does support ANSI
# C99's __func__, define __FUNCTION__ to __func__ and define
# HAVE__FUNC__ to 1.
AC_DEFUN([PYTHIA_CXX_FUNCTION], [
# $Id: pythia_cxx_function.m4,v 1.1 2005/09/09 16:12:02 leif Exp $
AC_MSG_CHECKING([if the compiler supports __FUNCTION__])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[[const char *foo = __FUNCTION__;]])],
    [AC_MSG_RESULT(yes)
     AC_DEFINE(HAVE__FUNC__, 1, [Define if the compiler supports __FUNCTION__.])],
    [AC_MSG_RESULT(no)
     AC_MSG_CHECKING([if the compiler supports __func__])
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[[const char *foo = __func__;]])],
        [AC_MSG_RESULT(yes)
         AC_DEFINE(__FUNCTION__, __func__, [Define to __func__ if the compiler supports __func__ but not __FUNCTION__.])
         AC_DEFINE(HAVE__FUNC__, 1, [Define to 1 if the compiler supports __FUNCTION__.])],
         [AC_MSG_RESULT(no)])
])
])dnl PYTHIA_CXX_FUNCTION
dnl end of file
