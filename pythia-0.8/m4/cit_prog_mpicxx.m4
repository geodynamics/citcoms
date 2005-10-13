# CIT_PROG_MPICXX
# ---------------
# Call AC_PROG_CXX, but prefer MPI C++ wrappers to a bare compiler in
# the search list.  Set MPICXX to the program/wrapper used to compile
# C++ MPI programs.  Set CXX to the compiler used to compile ordinary
# C++ programs, and link shared libraries of all types (see the
# comment about the MPI library, below).  Make sure that CXX and
# MPICXX both represent the same underlying C++ compiler.
AC_DEFUN([CIT_PROG_MPICXX], [
# $Id: cit_prog_mpicxx.m4,v 1.3 2005/09/30 18:09:01 leif Exp $
AC_BEFORE([$0], [AC_PROG_CXX])
AC_ARG_VAR(MPICXX, [MPI C++ compiler command])
AC_SUBST([MPICXX])
test -z "$want_mpi" && want_mpi=yes
# The 'cit_compiler_search_list' is the result of merging the
# following:
#     * MPI C++ wrappers
#     * the Autoconf default (g++ c++ gpp aCC CC cxx cc++ cl
#       FCC KCC RCC xlC_r xlC)
#     * the range of values for config's COMPILER_CXX_NAME (aCC CC cl
#       cxx ecpc g++ icpc KCC pgCC xlC xlc++_r xlC_r)
# Newer names are tried first (e.g., icpc before ecpc).
cit_compiler_search_list="g++ c++ gpp aCC CC cxx cc++ cl FCC KCC RCC xlc++_r xlC_r xlC"
cit_compiler_search_list="$cit_compiler_search_list icpc ecpc pgCC"
cit_mpicxx_search_list="mpicxx mpic++ mpiCC hcp mpCC mpxlC mpxlC_r cmpic++"
# There are two C++ command variables, so there are four cases to
# consider:
#
#     ./configure CXX=g++ MPICXX=mpicxx    # save MPICXX as cit_MPICXX; MPICXX=$CXX
#     ./configure CXX=g++                  # MPICXX=$CXX, guess cit_MPICXX
#     ./configure MPICXX=mpicxx            # derive CXX
#     ./configure                          # guess MPICXX and derive CXX
#
# In the cases where CXX is explicitly specified, the MPI C++ wrapper
# (cit_MPICXX, if known) is only used to gather compile/link flags (if
# needed).
if test "$want_mpi" = yes; then
    if test -n "$CXX"; then
        cit_MPICXX_underlying_CXX=$CXX
        if test -n "$MPICXX"; then
            # CXX=g++ MPICXX=mpicxx
            cit_MPICXX=$MPICXX
            MPICXX=$CXX
        else
            # CXX=g++ MPICXX=???
            AC_CHECK_PROGS(cit_MPICXX, $cit_mpicxx_search_list)
        fi
    else
        if test -n "$MPICXX"; then
            # CXX=??? MPICXX=mpicxx
            cit_MPICXX=$MPICXX
            CXX=$MPICXX # will be reevaluated below
        else
            # CXX=??? MPICXX=???
            cit_compiler_search_list="$cit_mpicxx_search_list $cit_compiler_search_list"
        fi
    fi
fi
AC_PROG_CXX($cit_compiler_search_list)
if test "$want_mpi" = yes; then
    if test -z "$MPICXX"; then
        MPICXX=$CXX
    fi
    if test -z "$cit_MPICXX"; then
        case $MPICXX in
            *mp* | hcp)
                cit_MPICXX=$MPICXX
                ;;
        esac
    fi
    # The MPI library is typically static.  Linking a shared object
    # against static library is non-portable, and needlessly bloats our
    # Python extension modules on the platforms where it does work.
    # Unless CXX was set explicitly, attempt to set CXX to the underlying
    # compiler command, so that we may link with the matching C++
    # compiler, but omit -lmpi/-lmpich from the link line.
    if test -z "$cit_MPICXX_underlying_CXX"; then
        if test -n "$cit_MPICXX"; then
            AC_MSG_CHECKING([for the C++ compiler underlying $cit_MPICXX])
            CXX=
            AC_LANG_PUSH(C++)
            # The variety of flags used by MPICH, LAM/MPI, Open MPI, and ChaMPIon/Pro.
            # NYI: mpxlC/mpCC (xlC?), mpxlC_r (xlC_r?)
            for cit_arg_show in "-show" "-showme" "-echo" "-compile_info"
            do
                cit_cmd="$cit_MPICXX -c $cit_arg_show"
                if $cit_cmd >/dev/null 2>&1; then
                    CXX=`$cit_cmd 2>/dev/null | sed 's/ .*//'`
                    if test -n "$CXX"; then
                        AC_COMPILE_IFELSE([AC_LANG_PROGRAM()], [break 2], [CXX=])
                    fi
                fi
            done
            AC_LANG_POP(C++)
            if test -n "$CXX"; then
                AC_MSG_RESULT($CXX)
            else
                AC_MSG_RESULT(failed)
                AC_MSG_FAILURE([can not determine the C++ compiler underlying $cit_MPICXX])
            fi
        fi
        cit_MPICXX_underlying_CXX=$CXX
    fi
fi
])dnl CIT_PROG_MPICXX
dnl end of file
