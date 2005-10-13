# CIT_PATH_PETSC([VERSION], [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -----------------------------------------------------------------------
# Check for the PETSc package.  Requires Python.
AC_DEFUN([CIT_PATH_PETSC], [
# $Id: cit_path_exchanger.m4 2367 2005-09-09 16:46:52Z leif $
AC_REQUIRE([AM_PATH_PYTHON])
AC_ARG_VAR(PETSC_DIR, [location of PETSc installation])
AC_ARG_VAR(PETSC_ARCH, [PETSc configuration])
AC_MSG_CHECKING([for PETSc dir])
if test -z "$PETSC_DIR"; then
    AC_MSG_RESULT(no)
    m4_default([$3], [AC_MSG_ERROR([PETSc not found; set PETSC_DIR])])
elif test ! -d "$PETSC_DIR"; then
    AC_MSG_RESULT(no)
    m4_default([$3], [AC_MSG_ERROR([PETSc not found; PETSC_DIR=$PETSC_DIR is invalid])])
elif test ! -d "$PETSC_DIR/include"; then
    m4_default([$3], [AC_MSG_ERROR([PETSc include dir $PETSC_DIR/include not found; check PETSC_DIR])])
elif test ! -f "$PETSC_DIR/include/petscversion.h"; then
    m4_default([$3], [AC_MSG_ERROR([PETSc header file $PETSC_DIR/include/petscversion.h not found; check PETSC_DIR])])
elif test -z "$PETSC_ARCH" && test ! -x "$PETSC_DIR/bin/configarch"; then
    m4_default([$3], [AC_MSG_ERROR([PETSc file $PETSC_DIR/bin/configarch not found; check PETSC_DIR])])
else
    AC_MSG_RESULT([$PETSC_DIR])
    AC_MSG_CHECKING([for PETSc arch])
    if test -z "$PETSC_ARCH"; then
        PETSC_ARCH=`$PETSC_DIR/bin/configarch`
    fi
    AC_MSG_RESULT([$PETSC_ARCH])
    if test ! -d "$PETSC_DIR/bmake/$PETSC_ARCH"; then
        m4_default([$3], [AC_MSG_ERROR([PETSc config dir $PETSC_DIR/bmake/$PETSC_ARCH not found; check PETSC_ARCH])])
    elif test ! -f "$PETSC_DIR/bmake/$PETSC_ARCH/petscconf"; then
        m4_default([$3], [AC_MSG_ERROR([PETSc config file $PETSC_DIR/bmake/$PETSC_ARCH/petscconf not found; check PETSC_ARCH])])
    else
        AC_MSG_CHECKING([for PETSc version == $1])
        echo "PETSC_DIR = $PETSC_DIR" > petscconf
        echo "PETSC_ARCH = $PETSC_ARCH" >> petscconf
        cat $PETSC_DIR/bmake/$PETSC_ARCH/petscconf $PETSC_DIR/bmake/common/variables >> petscconf
        cat >petsc.py <<END_OF_PYTHON
[from distutils.sysconfig import parse_config_h, parse_makefile, expand_makefile_vars

f = open('$PETSC_DIR/include/petscversion.h')
vars = parse_config_h(f)
f.close()

parse_makefile('petscconf', vars)

keys = (
    'PETSC_VERSION_MAJOR',
    'PETSC_VERSION_MINOR',
    'PETSC_VERSION_SUBMINOR',

    'PETSC_INCLUDE',
    'PETSC_LIB_DIR',
    'PETSC_LIB_BASIC',
    'PETSC_FORTRAN_LIB_BASIC',
    'PETSC_EXTERNAL_LIB_BASIC',

    'FC',
)

for key in keys:
    if key[:6] == 'PETSC_':
        print '%s="%s"' % (key, expand_makefile_vars(str(vars[key]), vars))
    else:
        print 'PETSC_%s="%s"' % (key, expand_makefile_vars(str(vars[key]), vars))

]
END_OF_PYTHON
        eval `$PYTHON petsc.py 2>/dev/null`
        rm -f petsc.py petscconf

        [eval `echo $1 | sed 's/\([^.]*\)[.]\([^.]*\).*/petsc_1_major=\1; petsc_1_minor=\2;/'`]
        if test -z "$PETSC_VERSION_MAJOR" -o -z "$PETSC_VERSION_MINOR"; then
            AC_MSG_RESULT(no)
            m4_default([$3], [AC_MSG_ERROR([no suitable PETSc package found])])
        elif test "$PETSC_VERSION_MAJOR" -eq "$petsc_1_major" -a \
                  "$PETSC_VERSION_MINOR" -eq "$petsc_1_minor" ; then
            AC_MSG_RESULT([yes ($PETSC_VERSION_MAJOR.$PETSC_VERSION_MINOR.$PETSC_VERSION_SUBMINOR)])
            $2
        else
            AC_MSG_RESULT([no ($PETSC_VERSION_MAJOR.$PETSC_VERSION_MINOR.$PETSC_VERSION_SUBMINOR)])
            m4_default([$3], [AC_MSG_ERROR([no suitable PETSc package found])])
        fi
    fi
fi
AC_SUBST([PETSC_VERSION_MAJOR])
AC_SUBST([PETSC_VERSION_MINOR])
AC_SUBST([PETSC_VERSION_SUBMINOR])
AC_SUBST([PETSC_INCLUDE])
AC_SUBST([PETSC_LIB_DIR])
AC_SUBST([PETSC_LIB_BASIC])
AC_SUBST([PETSC_FORTRAN_LIB_BASIC])
AC_SUBST([PETSC_EXTERNAL_LIB_BASIC])
AC_SUBST([PETSC_FC])
])dnl CIT_PATH_PETSC
dnl end of file
