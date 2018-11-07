# -*- Autoconf -*-

## --------------------------------------------------------- ##
## Autoconf macros for functions missing in older versions.  ##
## --------------------------------------------------------- ##

# Missing in autoconf < 2.60
m4_ifdef([AC_PROG_MKDIR_P], [], [
  AC_DEFUN([AC_PROG_MKDIR_P],
    [AC_REQUIRE([AM_PROG_MKDIR_P])dnl defined by automake
     MKDIR_P='$(mkdir_p)'
     AC_SUBST([MKDIR_P])])])


dnl End of file
