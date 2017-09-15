# -*- Autoconf -*-


## ---------------------------- ##
## Autoconf macros for Fortran. ##
## ---------------------------- ##


# CIT_FC_OPENMP_MODULE(FC, FCFLAGS)
# -----------------------------------------------------
AC_DEFUN([CIT_FC_OPENMP_MODULE], [
AC_LANG_PUSH(Fortran)
cit_fc_save_fc=$FC
cit_fc_save_fcflags=$FCFLAGS
FC=$1
FCFLAGS="$FCFLAGS $2"

AC_MSG_CHECKING([whether OpenMP directives work])

#AC_COMPILE_IFELSE(_CIT_FC_TRIVIAL_OPENMP_PROGRAM, [
#    AC_MSG_RESULT(yes)
#], [
#    AC_MSG_RESULT(no)
#    AC_MSG_FAILURE([cannot compile a trivial OpenMP program using $1])
#])

AC_LINK_IFELSE(_CIT_FC_TRIVIAL_OPENMP_PROGRAM, [
    AC_MSG_RESULT(yes)
], [
    AC_MSG_RESULT(no)
    AC_MSG_FAILURE([cannot link a trivial OpenMP program using $1 with flags: $2])
])

FC=$cit_fc_save_fc
FCFLAGS=$cit_fc_save_fcflags


AC_LANG_POP(Fortran)
])dnl CIT_FC_OPENMP_MODULE

AC_DEFUN([_CIT_FC_TRIVIAL_OPENMP_PROGRAM], [
AC_LANG_PROGRAM([], [[
       implicit none
       integer OMP_get_thread_num  
       integer OMP_GET_MAX_THREADS  
       integer NUM_THREADS
       integer thread_id
  
       NUM_THREADS = OMP_GET_MAX_THREADS()
       !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(thread_id)
       thread_id = OMP_get_thread_num()+1
       !$OMP END PARALLEL
]])
])dnl _CIT_FC_TRIVIAL_OPENMP_PROGRAM

dnl end of file
