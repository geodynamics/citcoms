# CIT_CHECK_LIB_MPI
# -----------------
AC_DEFUN([CIT_CHECK_LIB_MPI], [
# $Id: cit_check_lib_mpi.m4,v 1.4 2005/09/30 21:09:25 leif Exp $
AC_REQUIRE([_CIT_PROG_MPICC])dnl
AC_ARG_VAR(MPILIBS, [MPI linker flags, e.g. -L<mpi lib dir> -lmpi])
AC_SUBST(MPILIBS)
cit_save_CC=$CC
cit_save_CXX=$CXX
cit_save_LIBS=$LIBS
CC=$MPICC
CXX=$MPICXX
LIBS="$MPILIBS $LIBS"
# If MPILIBS is set, check to see if it works.
# If MPILIBS is not set, check to see if it is needed.
AC_CHECK_FUNC(MPI_Init, [], [
    if test -n "$MPILIBS"; then
        AC_MSG_ERROR([function MPI_Init not found; check MPILIBS])
    fi
    # MPILIBS is needed but was not set.
    AC_LANG_CASE(
        [C], [
            cit_mpicmd=$cit_MPICC
        ],
        [C++], [
            cit_mpicmd=$cit_MPICXX
            test -z "$cit_mpicmd" && cit_mpicmd=$cit_MPICC
        ]
    )
    cit_libs=
    if test -n "$cit_mpicmd"; then
        # Try to guess the correct value for MPILIBS using an MPI wrapper.
        AC_MSG_CHECKING([for the libraries used by $cit_mpicmd])
        for cit_arg_show in "-show" "-showme" "-echo" "-link_info"
        do
            cit_cmd="$cit_mpicmd $cit_arg_show"
            if $cit_cmd >/dev/null 2>&1; then
                cit_args=`$cit_cmd 2>/dev/null`
                test -z "$cit_args" && continue
                for cit_arg in $cit_args
                do
                    case $cit_arg in
                        -L* | -l* | -pthread*) cit_libs="$cit_libs $cit_arg" ;;
                    esac
                done
                test -z "$cit_libs" && continue
                break
            fi
        done
        if test -n "$cit_libs"; then
            AC_MSG_RESULT([$cit_libs])
            LIBS="$cit_libs $cit_save_LIBS"
            unset ac_cv_func_MPI_Init
            AC_CHECK_FUNC(MPI_Init, [
                MPILIBS=$cit_libs
                export MPILIBS
            ], [
                _CIT_CHECK_LIB_MPI_FAILED
            ])
        else
            AC_MSG_RESULT(failed)
        fi
    else
        # Desperate, last-ditch effort.
        for cit_lib in mpi mpich; do
            AC_CHECK_LIB($cit_lib, MPI_Init, [
                cit_libs="-l$cit_lib"
                MPILIBS=$cit_libs
                export MPILIBS
                break])
        done
        if test -z "$cit_libs"; then
            _CIT_CHECK_LIB_MPI_FAILED
        fi
    fi
])
LIBS=$cit_save_LIBS
CXX=$cit_save_CXX
CC=$cit_save_CC
])dnl CIT_CHECK_LIB_MPI

AC_DEFUN([_CIT_CHECK_LIB_MPI_FAILED], [
AC_MSG_ERROR([no MPI library found

    Set the MPICC, MPICXX, MPIINCLUDES, and MPILIBS environment variables
    to specify how to build MPI programs.
])
])dnl _CIT_CHECK_LIB_MPI_FAILED

dnl end of file
