##############################################################################
# -*- Autoconf -*-
#
# Implements the AM_OPTIONS_VTK and AM_PATH_VTK macros.
# The AM_OPTIONS_VTK macro adds the --with-vtk=path option,
# and the AM_PATH_VTK macro is used to detect VTK presence,
# location and version.
#
# Modified from http://www.vtk.org/Wiki/VTK_Autoconf
# Originally by Francesco Montorsi
#


##############################################################################
#
# CIT_OPTIONS_VTK
#
# Adds the --with-vtk=PATH option to the configure options
#
AC_DEFUN([CIT_OPTIONS_VTK],[

    # vtk install prefix
    AC_ARG_WITH([vtk],
                [AC_HELP_STRING(
                    [--with-vtk],
                    [The prefix where VTK is installed @<:@default=/usr@:>@])],
                [with_vtk="$withval"],
                [with_vtk="/usr"])

    # vtk suffix, to allow for multiple installations of vtk libraries
    AC_ARG_WITH([vtk-suffix],
                [AC_HELP_STRING(
                    [--with-vtk-suffix],
                    [Suffix to append to VTK's include directory, e.g., for vtk-5.2/ the suffix is "-5.2"])],
                [vtk_suffix="$withval"],
                [vtk_suffix=""])

    # alternatively, derive suffix from version number
    AC_ARG_WITH([vtk-version],
                [AC_HELP_STRING(
                    [--with-vtk-version],
                    [Version number of VTK library @<:@default=5.2@:>@])],
                [vtk_version="$withval"],
                [vtk_version="5.2"])

    if test -z "$vtk_suffix"; then
        #
        # suffix was not specified. use the version number
        # to calculate what it should be
        #
        if test -n "$vtk_version"; then
            vtk_suffix="-$vtk_version"
        fi

    else
        #
        # suffix was specified. ignore version silently?
        # produce error for now
        #
        if test -n "$vtk_version"; then
            AC_MSG_ERROR([The option --with-vtk-suffix overrides --with-vtk-version. Do not use simultaneously.])
        fi

    fi
])

##############################################################################
#
# CIT_PATH_VTK([minimum-version], [action-if-found], [action-if-not-found])
#
# NOTE: [minimum-version] must be in the form [X.Y.Z]
#
AC_DEFUN([CIT_PATH_VTK],[

    dnl do we want to check for VTK?
    if test x$with_vtk = "xyes"; then
        dnl in case user wrote --with-vtk=yes
        with_vtk="/usr"
    fi

    if test x$with_vtk != "xno"; then
        VTK_PREFIX="$with_vtk"

        AC_CHECK_FILE([$VTK_PREFIX/include/vtk$vtk_suffix/vtkCommonInstantiator.h],
                      [vtkFound="OK"])
        AC_MSG_CHECKING([if VTK is installed in $VTK_PREFIX])

        if test -z "$vtkFound"; then

            dnl VTK not found!
            AC_MSG_RESULT([no])
            $3

        else

            dnl VTK found!
            AC_MSG_RESULT([yes])

            dnl set VTK flags
            VTK_INCLUDES="-I$VTK_PREFIX/include/vtk$vtk_suffix"
            VTK_LDFLAGS="-L$VTK_PREFIX/lib/vtk$vtk_suffix -L$VTK_PREFIX/lib64/vtk$vtk_suffix"

            dnl remember the old flags
            AC_LANG_PUSH([C])
            OLD_CFLAGS=$CFLAGS
            OLD_CXXFLAGS=$CXXFLAGS
            OLD_LDFLAGS=$LDFLAGS
            OLD_LIBS=$LIBS

            dnl in order to be able to compile the following test programs,
            dnl we need to add to our VTK settings to the current flags
            CFLAGS="$VTK_CFLAGS $CFLAGS"
            CXXFLAGS="$VTK_CXXFLAGS $CXXFLAGS"
            LDFLAGS="$VTK_LDFLAGS $LDFLAGS"

            dnl these are the VTK libraries of a default build.
            dnl figure out vtkCommon, vtkIO, vtkFiltering, plus dependencies (in case VTK libs are static)
            dnl order of libs is significant
            VTK_SUPPORT_LIBS="-lvtktiff -lvtkpng -lvtkjpeg -lvtkzlib -lvtkexpat -lvfw32 -lgdi32"
            AC_CHECK_LIB(vtkIO, strcmp, [], [
                VTK_SUPPORT_LIBS="-lvtktiff -lvtkpng -lvtkjpeg -lvtkzlib -lvtkexpat"
                AC_CHECK_LIB(vtkIO, abort, [], [
                    VTK_SUPPORT_LIBS="-ltiff -lpng -ljpeg -lz -lexpat"
                    AC_CHECK_LIB(vtkIO, exit, [], [
                        VTK_SUPPORT_LIBS=""
                        AC_CHECK_LIB(vtkIO, strstr, [], [
                            AC_MSG_ERROR([cannot link against VTK libraries])
                        ], [$VTK_SUPPORT_LIBS])
                    ], [$VTK_SUPPORT_LIBS])
                ], [$VTK_SUPPORT_LIBS])
            ], [$VTK_SUPPORT_LIBS])
            VTK_LIBS="-lvtkIO -lvtkDICOMParser -lvtkFiltering -lvtkGenericFiltering -lvtkCommon $VTK_SUPPORT_LIBS -lvtksys"
            LIBS="$VTK_LIBS $LIBS"

            dnl now, check that we don't exceed the specified version
            if test -n "$1"; then
                dnl the version of VTK we need

                maj=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
                min=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
                rel=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`
                AC_MSG_CHECKING([if VTK version is at least $maj.$min.$rel])

                dnl check if the installed VTK is greater or not
                AC_COMPILE_IFELSE([
                    AC_LANG_PROGRAM([[
                        #include <vtk/vtkConfigure.h>
                        #include <stdio.h>
                        ]],[[
                        printf("VTK version is: %d.%d.%d",
                               VTK_MAJOR_VERSION,
                               VTK_MINOR_VERSION,
                               VTK_BUILD_VERSION);

                        #if VTK_MAJOR_VERSION < $maj
                        #error Installed VTK is too old!
                        #endif

                        #if VTK_MINOR_VERSION < $min
                        #error Installed VTK is too old!
                        #endif

                        #if VTK_BUILD_VERSION < $rel
                        #error Installed VTK is too old!
                        #endif
                    ]])
                ], [vtkVersion="OK"])

                dnl restore all flags without VTK values
                CFLAGS=$OLD_CFLAGS
                CXXFLAGS=$OLD_CXXFLAGS
                LDFLAGS=$OLD_LDFLAGS
                LIBS=$OLD_LIBS

                if test "$vtkVersion" = "OK"; then
                    AC_MSG_RESULT([yes])
                    $2
                else
                    AC_MSG_RESULT([no])
                    $3
                fi

            else

                dnl restore all flags without VTK values
                CFLAGS=$OLD_CFLAGS
                CXXFLAGS=$OLD_CXXFLAGS
                LDFLAGS=$OLD_LDFLAGS
                LIBS=$OLD_LIBS

                dnl if we don't have to check for minimum version
                dnl (because the user did not set that option),
                dnl then we can execute here the block action-if-found
                $2

            fi          #if [[ -n "$1" ]];

            dnl Don't move this up to where we restore flags, because it doesn't work.
            AC_LANG_POP([C])

            dnl Finally, define the C preprocessor variable HAVE_VTK
            AC_DEFINE([HAVE_VTK],,[define if the VTK library is available])

        fi          # if [[ -z "$vtkFound" ]];
    fi          # $with_vtk != "no"

    dnl Substitute the values of our VTK flags
    AC_SUBST(VTK_INCLUDES)
    AC_SUBST(VTK_LDFLAGS)
    AC_SUBST(VTK_LIBS)
])



# vim: syntax=config
