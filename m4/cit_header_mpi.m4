# CIT_HEADER_MPI
# --------------
AC_DEFUN([CIT_HEADER_MPI], [
# $Id$
AC_REQUIRE([_CIT_PROG_MPICC])dnl
AC_ARG_VAR(MPIINCLUDES, [MPI include flags, e.g. -I<mpi include dir>])
AC_SUBST(MPIINCLUDES)
cit_save_CC=$CC
cit_save_CXX=$CXX
cit_save_CPPFLAGS=$CPPFLAGS
CC=$MPICC
CXX=$MPICXX
CPPFLAGS="$MPIINCLUDES $CPPFLAGS"
# If MPIINCLUDES is set, check to see if it works.
# If MPIINCLUDES is not set, check to see if it is needed.
AC_MSG_CHECKING([for mpi.h])
dnl Use AC_TRY_COMPILE instead of AC_CHECK_HEADER because the
dnl latter also preprocesses using $CXXCPP.
AC_TRY_COMPILE([
#include <mpi.h>
], [], [
    AC_MSG_RESULT(yes)
], [
    AC_MSG_RESULT(no)
    if test -n "$MPIINCLUDES"; then
        AC_MSG_ERROR([header <mpi.h> not found; check MPIINCLUDES])
    fi
    # MPIINCLUDES is needed but was not set.
    AC_LANG_CASE(
        [C], [
            cit_mpicmd=$cit_MPICC
        ],
        [C++], [
            cit_mpicmd=$cit_MPICXX
            test -z "$cit_mpicmd" && cit_mpicmd=$cit_MPICC
        ]
    )
    if test -n "$cit_mpicmd"; then
        # Try to guess the correct value for MPIINCLUDES using an MPI wrapper.
        AC_MSG_CHECKING([for the includes used by $cit_mpicmd])
        cit_includes=
        for cit_arg_show in "-show" "-showme" "-echo" "-compile_info"
        do
            cit_cmd="$cit_mpicmd -c $cit_arg_show"
            if $cit_cmd >/dev/null 2>&1; then
                cit_args=`$cit_cmd 2>/dev/null`
                test -z "$cit_args" && continue
                for cit_arg in $cit_args
                do
                    case $cit_arg in
                        -I*) cit_includes="$cit_includes $cit_arg" ;;
                    esac
                done
                test -z "$cit_includes" && continue
                break
            fi
        done
        if test -n "$cit_includes"; then
            AC_MSG_RESULT([$cit_includes])
            AC_MSG_CHECKING([for mpi.h])
            CPPFLAGS="$cit_includes $cit_save_CPPFLAGS"
            AC_TRY_COMPILE([
#include <mpi.h>
            ], [], [
                AC_MSG_RESULT(yes)
                MPIINCLUDES=$cit_includes
                export MPIINCLUDES
            ], [
                AC_MSG_RESULT(no)
                _CIT_HEADER_MPI_FAILED
            ])
        else
            AC_MSG_RESULT(failed)
            _CIT_HEADER_MPI_FAILED
        fi
    else
        _CIT_HEADER_MPI_FAILED
    fi
])
# If the current language is C++, verify that the MPI library is
# link-compatible with CXX (which could be different than the C++
# compiler used to build the MPI library).  If there is a problem,
# attempt to work-around it by preventing MPI's C++ bindings from
# being #included.
AC_LANG_CASE([C++], [
    AC_MSG_CHECKING([whether we can link a trivial C++ MPI program])
    cit_save_LIBS=$LIBS
    LIBS="$MPILIBS $LIBS"
    CPPFLAGS="$MPIINCLUDES $cit_save_CPPFLAGS"
    AC_LINK_IFELSE([_CIT_TRIVIAL_MPI_PROGRAM], [AC_MSG_RESULT(yes)], [
        AC_MSG_RESULT(no)
        cit_status="broken"
        for cit_arg in "-DMPICH_SKIP_MPICXX" "-UHAVE_MPI_CPP" "-DLAM_WANT_MPI2CPP=0" "-DLAM_BUILDING=1" "-DOMPI_WANT_CXX_BINDINGS=0" "-DOMPI_BUILDING=1"
        do
            CPPFLAGS="$cit_arg $MPIINCLUDES $cit_save_CPPFLAGS"
            AC_MSG_CHECKING([whether we can link a trivial C++ MPI program with $cit_arg])
            AC_LINK_IFELSE([_CIT_TRIVIAL_MPI_PROGRAM], [
                AC_MSG_RESULT(yes)
                MPIINCLUDES="$cit_arg $MPIINCLUDES"
                export MPIINCLUDES
                cit_status="fixed"
                break
            ], [
                AC_MSG_RESULT(no)
            ])
        done
        if test "$cit_status" = "broken"; then
            AC_MSG_FAILURE([cannot link a trivial C++ MPI program using $CXX])
        fi
    ])
    LIBS=$cit_save_LIBS
])
CPPFLAGS=$cit_save_CPPFLAGS
CXX=$cit_save_CXX
CC=$cit_save_CC
])dnl CIT_HEADER_MPI

AC_DEFUN([_CIT_HEADER_MPI_FAILED], [
AC_MSG_ERROR([header <mpi.h> not found

    Set the MPICC, MPICXX, MPIINCLUDES, and MPILIBS environment variables
    to specify how to build MPI programs.
])
])dnl CIT_HEADER_MPI_FAILED

AC_DEFUN([_CIT_TRIVIAL_MPI_PROGRAM], [
AC_LANG_PROGRAM([[
#include <mpi.h>
]], [[
    MPI_Init(0,0);
    MPI_Finalize();
]])
])dnl _CIT_TRIVIAL_MPI_PROGRAM

dnl end of file
