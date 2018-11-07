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


#
# The following two macros are from autoconf 2.68 (which is still new).
#

# AC_FC_PP_SRCEXT(EXT, [ACTION-IF-SUCCESS], [ACTION-IF-FAILURE])
# --------------------------------------------------------------
# Like AC_FC_SRCEXT, set the source-code extension used in Fortran (FC) tests
# to EXT (which defaults to f).  Also, look for any necessary additional
# FCFLAGS needed to allow this extension for preprocessed Fortran, and store
# them in the output variable FCFLAGS_<EXT> (e.g. FCFLAGS_f90 for EXT=f90).
# If successful, call ACTION-IF-SUCCESS.  If unable to compile preprocessed
# source code with EXT, call ACTION-IF-FAILURE, which defaults to failing with
# an error message.
#
# Some compilers allow preprocessing with either a Fortran preprocessor or
# with the C preprocessor (cpp).  Prefer the Fortran preprocessor, to deal
# correctly with continuation lines, `//' (not a comment), and preserve white
# space (for fixed form).
#
# (The flags for the current source-code extension, if any, are stored in
# $ac_fcflags_srcext and used automatically in subsequent autoconf tests.)
#
# For ordinary extensions like f90, etcetera, the modified FCFLAGS
# are needed for IBM's xlf*.  Also, for Intel's ifort compiler, the
# $FCFLAGS_<EXT> variable *must* go immediately before the source file on the
# command line, unlike other $FCFLAGS.  Ugh.
#
# Known extensions that enable preprocessing by default, and flags to force it:
# GNU: .F .F90 .F95 .F03 .F08, -cpp for most others,
#      -x f77-cpp-input for .f77 .F77; -x f95-cpp-input for gfortran < 4.4
# SGI: .F .F90, -ftpp or -cpp for .f .f90, -E write preproc to stdout
#      -macro_expand enable macro expansion everywhere (with -ftpp)
#      -P preproc only, save in .i, no #line's
# SUN: .F .F95, -fpp for others; -xpp={fpp,cpp} for preprocessor selection
#      -F preprocess only (save in lowercase extension)
# IBM: .F .F77 .F90 .F95 .F03, -qsuffix=cpp=EXT for extension .EXT to invoke cpp
#      -WF,-qnofpp -WF,-qfpp=comment:linecont:nocomment:nolinecont
#      -WF,-qlanglvl=classic or not -qnoescape (trigraph problems)
#      -d no #line in output, -qnoobject for preprocessing only (output in .f)
#      -q{no,}ppsuborigarg substitute original macro args before expansion
# HP:  .F, +cpp={yes|no|default} use cpp, -cpp, +cpp_keep save in .i/.i90
# PGI: -Mpreprocess
# Absoft: .F .FOR .F90 .F95, -cpp for others
# Cray: .F .F90 .FTN, -e Z for others; -F enable macro expansion everywhere
# Intel: .F .F90, -fpp for others, but except for .f and .f90, -Tf may also be
#        needed right before the source file name
# PathScale: .F .F90 .F95, -ftpp or -cpp for .f .f90 .f95
#         -macro_expand for expansion everywhere, -P for no #line in output
# Lahey: .F .FOR .F90 .F95, -Cpp
# NAGWare: .F .F90 .F95, .ff .ff90 .ff95 (new), -fpp for others
# Compaq/Tru64: .F .F90, -cpp, -P keep .i file, -P keep .i file
# f2c: .F, -cpp
# g95: .F .FOR .F90 .F95 .F03, -cpp -no-cpp, -E for stdout
m4_version_prereq([2.68], [], [
AC_DEFUN([AC_FC_PP_SRCEXT],
[AC_LANG_PUSH(Fortran)dnl
AC_CACHE_CHECK([for Fortran flag to compile preprocessed .$1 files],
		ac_cv_fc_pp_srcext_$1,
[ac_ext=$1
ac_fcflags_pp_srcext_save=$ac_fcflags_srcext
ac_fcflags_srcext=
ac_cv_fc_pp_srcext_$1=unknown
case $ac_ext in #(
  [[fF]]77) ac_try=f77-cpp-input;; #(
  *) ac_try=f95-cpp-input;;
esac
for ac_flag in none -ftpp -fpp -Tf "-fpp -Tf" -xpp=fpp -Mpreprocess "-e Z" \
               -cpp -xpp=cpp -qsuffix=cpp=$1 "-x $ac_try" +cpp -Cpp; do
  test "x$ac_flag" != xnone && ac_fcflags_srcext="$ac_flag"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [[
#if 0
#include <ac_nonexistent.h>
      choke me
#endif]])],
    [AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [[
#if 1
#include <ac_nonexistent.h>
      choke me
#endif]])],
       [],
       [ac_cv_fc_pp_srcext_$1=$ac_flag; break])])
done
rm -f conftest.$ac_objext conftest.$1
ac_fcflags_srcext=$ac_fcflags_pp_srcext_save
])
if test "x$ac_cv_fc_pp_srcext_$1" = xunknown; then
  m4_default([$3],
             [AC_MSG_ERROR([Fortran could not compile preprocessed .$1 files])])
else
  ac_fc_srcext=$1
  if test "x$ac_cv_fc_pp_srcext_$1" = xnone; then
    ac_fcflags_srcext=""
    FCFLAGS_[]$1[]=""
  else
    ac_fcflags_srcext=$ac_cv_fc_pp_srcext_$1
    FCFLAGS_[]$1[]=$ac_cv_fc_pp_srcext_$1
  fi
  AC_SUBST(FCFLAGS_[]$1)
  $2
fi
AC_LANG_POP(Fortran)dnl
])# AC_FC_PP_SRCEXT
])

# AC_FC_PP_DEFINE([ACTION-IF-SUCCESS], [ACTION-IF-FAILURE = FAILURE])
# -------------------------------------------------------------------
# Find a flag to specify defines for preprocessed Fortran.  Not all
# Fortran compilers use -D.  Substitute FC_DEFINE with the result and
# call ACTION-IF-SUCCESS (defaults to nothing) if successful, and
# ACTION-IF-FAILURE (defaults to failing with an error message) if not.
#
# Known flags:
# IBM: -WF,-D
# Lahey/Fujitsu: -Wp,-D     older versions???
# f2c: -D or -Wc,-D
# others: -D
m4_version_prereq([2.68], [], [
AC_DEFUN([AC_FC_PP_DEFINE],
[AC_LANG_PUSH([Fortran])dnl
ac_fc_pp_define_srcext_save=$ac_fc_srcext
AC_FC_PP_SRCEXT([F])
AC_CACHE_CHECK([how to define symbols for preprocessed Fortran],
  [ac_cv_fc_pp_define],
[ac_fc_pp_define_srcext_save=$ac_fc_srcext
ac_cv_fc_pp_define=unknown
ac_fc_pp_define_FCFLAGS_save=$FCFLAGS
for ac_flag in -D -WF,-D -Wp,-D -Wc,-D
do
  FCFLAGS="$ac_fc_pp_define_FCFLAGS_save ${ac_flag}FOOBAR ${ac_flag}ZORK=42"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [[
#ifndef FOOBAR
      choke me
#endif
#if ZORK != 42
      choke me
#endif]])],
    [ac_cv_fc_pp_define=$ac_flag])
  test x"$ac_cv_fc_pp_define" != xunknown && break
done
FCFLAGS=$ac_fc_pp_define_FCFLAGS_save
])
ac_fc_srcext=$ac_fc_pp_define_srcext_save
if test "x$ac_cv_fc_pp_define" = xunknown; then
  FC_DEFINE=
  m4_default([$2],
	     [AC_MSG_ERROR([Fortran does not allow to define preprocessor symbols], 77)])
else
  FC_DEFINE=$ac_cv_fc_pp_define
  $1
fi
AC_SUBST([FC_DEFINE])dnl
AC_LANG_POP([Fortran])dnl
])
])


#
# The following three macros are from autoconf 2.69 (which is quite new).
#

# AC_FC_MODULE_EXTENSION
# ----------------------
# Find the Fortran 90 module file extension.  The module extension is stored
# in the variable FC_MODEXT and empty if it cannot be determined.  The result
# or "unknown" is cached in the cache variable ac_cv_fc_module_ext.
m4_version_prereq([2.69], [], [
AC_DEFUN([AC_FC_MODULE_EXTENSION],
[AC_CACHE_CHECK([Fortran 90 module extension], [ac_cv_fc_module_ext],
[AC_LANG_PUSH([Fortran])
mkdir conftest.dir
cd conftest.dir
ac_cv_fc_module_ext=unknown
AC_COMPILE_IFELSE([[
      module conftest_module
      contains
      subroutine conftest_routine
      write(*,'(a)') 'gotcha!'
      end subroutine
      end module]],
  [ac_cv_fc_module_ext=`ls | sed -n 's,conftest_module\.,,p'`
   if test x$ac_cv_fc_module_ext = x; then
dnl Some F90 compilers use upper case characters for the module file name.
     ac_cv_fc_module_ext=`ls | sed -n 's,CONFTEST_MODULE\.,,p'`
   fi])
cd ..
rm -rf conftest.dir
AC_LANG_POP([Fortran])
])
FC_MODEXT=$ac_cv_fc_module_ext
if test "$FC_MODEXT" = unknown; then
  FC_MODEXT=
fi
AC_SUBST([FC_MODEXT])dnl
])
])


# AC_FC_MODULE_FLAG([ACTION-IF-SUCCESS], [ACTION-IF-FAILURE = FAILURE])
# ---------------------------------------------------------------------
# Find a flag to include Fortran 90 modules from another directory.
# If successful, run ACTION-IF-SUCCESS (defaults to nothing), otherwise
# run ACTION-IF-FAILURE (defaults to failing with an error message).
# The module flag is cached in the ac_cv_fc_module_flag variable.
# It may contain significant trailing whitespace.
#
# Known flags:
# gfortran: -Idir, -I dir (-M dir, -Mdir (deprecated), -Jdir for writing)
# g95: -I dir (-fmod=dir for writing)
# SUN: -Mdir, -M dir (-moddir=dir for writing;
#                     -Idir for includes is also searched)
# HP: -Idir, -I dir (+moddir=dir for writing)
# IBM: -Idir (-qmoddir=dir for writing)
# Intel: -Idir -I dir (-mod dir for writing)
# Absoft: -pdir
# Lahey: -mod dir
# Cray: -module dir, -p dir (-J dir for writing)
#       -e m is needed to enable writing .mod files at all
# Compaq: -Idir
# NAGWare: -I dir
# PathScale: -I dir  (but -module dir is looked at first)
# Portland: -module dir (first -module also names dir for writing)
# Fujitsu: -Am -Idir (-Mdir for writing is searched first, then '.', then -I)
#                    (-Am indicates how module information is saved)
m4_version_prereq([2.69], [], [
AC_DEFUN([AC_FC_MODULE_FLAG],[
AC_CACHE_CHECK([Fortran 90 module inclusion flag], [ac_cv_fc_module_flag],
[AC_LANG_PUSH([Fortran])
ac_cv_fc_module_flag=unknown
mkdir conftest.dir
cd conftest.dir
AC_COMPILE_IFELSE([[
      module conftest_module
      contains
      subroutine conftest_routine
      write(*,'(a)') 'gotcha!'
      end subroutine
      end module]],
  [cd ..
   ac_fc_module_flag_FCFLAGS_save=$FCFLAGS
   # Flag ordering is significant for gfortran and Sun.
   for ac_flag in -M -I '-I ' '-M ' -p '-mod ' '-module ' '-Am -I'; do
     # Add the flag twice to prevent matching an output flag.
     FCFLAGS="$ac_fc_module_flag_FCFLAGS_save ${ac_flag}conftest.dir ${ac_flag}conftest.dir"
     AC_COMPILE_IFELSE([[
      program main
      use conftest_module
      call conftest_routine
      end program]],
       [ac_cv_fc_module_flag="$ac_flag"])
     if test "$ac_cv_fc_module_flag" != unknown; then
       break
     fi
   done
   FCFLAGS=$ac_fc_module_flag_FCFLAGS_save
])
rm -rf conftest.dir
AC_LANG_POP([Fortran])
])
if test "$ac_cv_fc_module_flag" != unknown; then
  FC_MODINC=$ac_cv_fc_module_flag
  $1
else
  FC_MODINC=
  m4_default([$2],
    [AC_MSG_ERROR([unable to find compiler flag for module search path])])
fi
AC_SUBST([FC_MODINC])
# Ensure trailing whitespace is preserved in a Makefile.
AC_SUBST([ac_empty], [""])
AC_CONFIG_COMMANDS_PRE([case $FC_MODINC in #(
  *\ ) FC_MODINC=$FC_MODINC'${ac_empty}' ;;
esac])dnl
])
])


# AC_FC_MODULE_OUTPUT_FLAG([ACTION-IF-SUCCESS], [ACTION-IF-FAILURE = FAILURE])
# ----------------------------------------------------------------------------
# Find a flag to write Fortran 90 module information to another directory.
# If successful, run ACTION-IF-SUCCESS (defaults to nothing), otherwise
# run ACTION-IF-FAILURE (defaults to failing with an error message).
# The module flag is cached in the ac_cv_fc_module_output_flag variable.
# It may contain significant trailing whitespace.
#
# For known flags, see the documentation of AC_FC_MODULE_FLAG above.
m4_version_prereq([2.69], [], [
AC_DEFUN([AC_FC_MODULE_OUTPUT_FLAG],[
AC_CACHE_CHECK([Fortran 90 module output flag], [ac_cv_fc_module_output_flag],
[AC_LANG_PUSH([Fortran])
mkdir conftest.dir conftest.dir/sub
cd conftest.dir
ac_cv_fc_module_output_flag=unknown
ac_fc_module_output_flag_FCFLAGS_save=$FCFLAGS
# Flag ordering is significant: put flags late which some compilers use
# for the search path.
for ac_flag in -J '-J ' -fmod= -moddir= +moddir= -qmoddir= '-mod ' \
	      '-module ' -M '-Am -M' '-e m -J '; do
  FCFLAGS="$ac_fc_module_output_flag_FCFLAGS_save ${ac_flag}sub"
  AC_COMPILE_IFELSE([[
      module conftest_module
      contains
      subroutine conftest_routine
      write(*,'(a)') 'gotcha!'
      end subroutine
      end module]],
    [cd sub
     AC_COMPILE_IFELSE([[
      program main
      use conftest_module
      call conftest_routine
      end program]],
       [ac_cv_fc_module_output_flag="$ac_flag"])
     cd ..
     if test "$ac_cv_fc_module_output_flag" != unknown; then
       break
     fi])
done
FCFLAGS=$ac_fc_module_output_flag_FCFLAGS_save
cd ..
rm -rf conftest.dir
AC_LANG_POP([Fortran])
])
if test "$ac_cv_fc_module_output_flag" != unknown; then
  FC_MODOUT=$ac_cv_fc_module_output_flag
  $1
else
  FC_MODOUT=
  m4_default([$2],
    [AC_MSG_ERROR([unable to find compiler flag to write module information to])])
fi
AC_SUBST([FC_MODOUT])
# Ensure trailing whitespace is preserved in a Makefile.
AC_SUBST([ac_empty], [""])
AC_CONFIG_COMMANDS_PRE([case $FC_MODOUT in #(
  *\ ) FC_MODOUT=$FC_MODOUT'${ac_empty}' ;;
esac])dnl
])
])


dnl End of file
