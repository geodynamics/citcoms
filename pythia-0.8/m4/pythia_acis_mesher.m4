# PYTHIA_ACIS_MESHER
# ------------------
# NYI
AC_DEFUN([PYTHIA_ACIS_MESHER], [
# $Id: pythia_acis_mesher.m4,v 1.1 2005/09/09 16:12:02 leif Exp $
AC_DEFINE(ACIS_HAS_MESHER,,[Define if ACIS has api_initialize_mesh_surfaces().])
AM_CONDITIONAL([COND_ACIS_HAS_MESHER], [false])
])dnl PYTHIA_ACIS_MESHER
dnl end of file
