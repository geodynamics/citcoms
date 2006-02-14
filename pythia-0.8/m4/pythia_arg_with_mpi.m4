# PYTHIA_ARG_WITH_MPI
# -------------------
AC_DEFUN([PYTHIA_ARG_WITH_MPI], [
# $Id: pythia_arg_with_mpi.m4,v 1.1 2005/09/09 16:12:02 leif Exp $
AC_ARG_WITH([mpi],
    [AC_HELP_STRING([--with-mpi],
        [build with Message Passing Interface support @<:@default=yes@:>@])],
    [want_mpi="$withval"],
    [want_mpi=yes])
AM_CONDITIONAL([COND_MPI], [test "$want_mpi" = yes])
])dnl PYTHIA_ARG_WITH_MPI
dnl end of file
