# CIT_PROG_MPICC
# --------------
# Call AC_PROG_CC, but prefer MPI C wrappers to a bare compiler in
# the search list.  Set MPICC to the program/wrapper used to compile
# C MPI programs.  Set CC to the compiler used to compile ordinary
# C programs, and link shared libraries of all types (see the
# comment about the MPI library, below).  Make sure that CC and
# MPICC both represent the same underlying C compiler.
AC_DEFUN([CIT_PROG_MPICC], [
AC_PROVIDE([_CIT_PROG_MPICC])dnl
AC_REQUIRE([_CIT_PROG_MPICC_SEARCH_LIST])dnl
AC_BEFORE([$0], [AC_PROG_CC])
AC_ARG_VAR(MPICC, [MPI C compiler command])
AC_SUBST([MPICC])
test -z "$want_mpi" && want_mpi=yes
# The 'cit_compiler_search_list' is the result of merging the
# following:
#     * MPI C wrappers
#     * the range of values for config's COMPILER_CC_NAME
#       (cc cl ecc gcc icc pgcc xlc xlc_r)
# Newer names are tried first (e.g., icc before ecc).
cit_compiler_search_list="gcc cc cl icc ecc pgcc xlc xlc_r"
# There are two C command variables, so there are four cases to
# consider:
#
#     ./configure CC=gcc MPICC=mpicc       # save MPICC as cit_MPICC; MPICC=$CC
#     ./configure CC=gcc                   # MPICC=$CC, guess cit_MPICC
#     ./configure MPICC=mpicc              # derive CC
#     ./configure                          # guess MPICC and derive CC
#
# In the cases where CC is explicitly specified, the MPI C wrapper
# (cit_MPICC, if known) is only used to gather compile/link flags (if
# needed).
if test "$want_mpi" = yes; then
    if test -n "$CC"; then
        cit_MPICC_underlying_CC=$CC
        if test -n "$MPICC"; then
            # CC=gcc MPICC=mpicc
            cit_MPICC=$MPICC
            MPICC=$CC
        else
            # CC=gcc MPICC=???
            AC_CHECK_PROGS(cit_MPICC, $cit_mpicc_search_list)
        fi
    else
        if test -n "$MPICC"; then
            # CC=??? MPICC=mpicc
            cit_MPICC=$MPICC
            CC=$MPICC # will be reevaluated below
        else
            # CC=??? MPICC=???
            cit_compiler_search_list="$cit_mpicc_search_list $cit_compiler_search_list"
        fi
    fi
fi
AC_PROG_CC($cit_compiler_search_list)
if test "$want_mpi" = yes; then
    if test -z "$MPICC"; then
        MPICC=$CC
    fi
    if test -z "$cit_MPICC"; then
        case $MPICC in
            *mp* | hcc)
                cit_MPICC=$MPICC
                ;;
        esac
    fi
    # The MPI library is typically static.  Linking a shared object
    # against static library is non-portable, and needlessly bloats our
    # Python extension modules on the platforms where it does work.
    # Unless CC was set explicitly, attempt to set CC to the underlying
    # compiler command, so that we may link with the matching C
    # compiler, but omit -lmpi/-lmpich from the link line.
    if test -z "$cit_MPICC_underlying_CC"; then
        if test -n "$cit_MPICC"; then
            AC_MSG_CHECKING([for the C compiler underlying $cit_MPICC])
            CC=
            AC_LANG_PUSH(C)
            # The variety of flags used by MPICH, LAM/MPI, Open MPI, and ChaMPIon/Pro.
            # NYI: mpxlc/mpcc (xlc?), mpcc_r (xlc_r?)
            for cit_arg_show in "-show" "-showme" "-echo" "-compile_info"
            do
                cit_cmd="$cit_MPICC -c $cit_arg_show"
                if $cit_cmd >/dev/null 2>&1; then
                    CC=`$cit_cmd 2>/dev/null | sed 's/ .*//'`
                    if test -n "$CC"; then
                        AC_COMPILE_IFELSE([AC_LANG_PROGRAM()], [break 2], [CC=])
                    fi
                fi
            done
            AC_LANG_POP(C)
            if test -n "$CC"; then
                AC_MSG_RESULT($CC)
            else
                AC_MSG_RESULT(failed)
                AC_MSG_FAILURE([can not determine the C compiler underlying $cit_MPICC])
            fi
        fi
        cit_MPICC_underlying_CC=$CC
    fi
fi
])dnl CIT_PROG_MPICC

# _CIT_PROG_MPICC
# ---------------
# Search for an MPI C wrapper. ~ This private macro is employed by
# C++-only projects (via CIT_CHECK_LIB_MPI and CIT_HEADER_MPI).  It
# handles the case where an MPI C wrapper is present, but an MPI C++
# wrapper is missing or broken.  This can happen if a C++ compiler was
# not found/specified when MPI was installed.
AC_DEFUN([_CIT_PROG_MPICC], [
AC_REQUIRE([_CIT_PROG_MPICC_SEARCH_LIST])dnl
AC_CHECK_PROGS(cit_MPICC, $cit_mpicc_search_list)
])dnl _CIT_PROG_MPICC

AC_DEFUN([_CIT_PROG_MPICC_SEARCH_LIST], [
# $Id$
cit_mpicc_search_list="mpicc hcc mpcc mpcc_r mpxlc cmpicc"
])dnl _CIT_PROG_MPICC_SEARCH_LIST

dnl end of file
