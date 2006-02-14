# PYTHIA_ARG_WITH_ACIS
# --------------------
AC_DEFUN([PYTHIA_ARG_WITH_ACIS], [
# $Id: pythia_arg_with_acis.m4,v 1.1 2005/09/09 16:12:02 leif Exp $
AC_ARG_WITH([acis],
    [AC_HELP_STRING([--with-acis],
        [build with support for Spatial's 3D ACIS Modeler @<:@default=no@:>@])],
    [want_acis="$withval"],
    [want_acis=no])
AM_CONDITIONAL([COND_ACIS], [test "$want_acis" = yes])
])dnl PYTHIA_ARG_WITH_ACIS
dnl end of file
