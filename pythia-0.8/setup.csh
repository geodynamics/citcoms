#!/bin/csh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Generic tools

  setenv TOOLS_DIR /home/tools
  setenv TOOLS_INCDIR ${TOOLS_DIR}/include
  setenv TOOLS_LIBDIR ${TOOLS_DIR}/lib
  setenv LD_LIBRARY_PATH ${TOOLS_DIR}/lib
  setenv PATH ${TOOLS_DIR}/bin:${PATH}

# local geometry

  setenv PYTHIA_VERSION 0.8

  setenv DV_DIR ${HOME}/dv
  setenv PYTHIA_DIR ${DV_DIR}/pythia-${PYTHIA_VERSION}
  setenv TEMPLATES_DIR ${DV_DIR}/templates

  setenv BLD_ROOT ${DV_DIR}/builds/pythia-${PYTHIA_VERSION}
  setenv EXPORT_ROOT ${TOOLS_DIR}/pythia-${PYTHIA_VERSION}

  setenv PATH ${PATH}:${EXPORT_ROOT}/bin
  setenv PYTHONPATH ${EXPORT_ROOT}/modules
  setenv LD_LIBRARY_PATH ${EXPORT_ROOT}/lib:${LD_LIBRARY_PATH}

# build system

  setenv GNU_MAKE make
  setenv DEVELOPER ${USER}
  setenv TARGET shared,mpi,debug
  setenv TARGET_F77 Absoft-8.0

  setenv BLD_CONFIG ${DV_DIR}/config
  setenv PATH ${BLD_CONFIG}/make:${PATH}

# Python

  setenv PYTHON_VERSION 2.3
  setenv PYTHON_DIR /usr
  setenv PYTHON_LIBDIR ${PYTHON_DIR}/lib/python${PYTHON_VERSION}
  setenv PYTHON_INCDIR ${PYTHON_DIR}/include/python${PYTHON_VERSION}
  setenv PYTHONSTARTUP ${HOME}/.python
  setenv PATH ${PYTHON_DIR}/bin:${PATH}

# CVS

  setenv CVS_RSH ssh

# mpi

  setenv MPI_VERSION 1.2.5
  setenv MPI_DIR ${TOOLS_DIR}/mpich-${MPI_VERSION}_absoft8
  setenv MPI_INCDIR ${MPI_DIR}/include
  setenv MPI_LIBDIR ${MPI_DIR}/lib
  setenv PATH ${MPI_DIR}/bin:${PATH}

# version
# $Id: setup.csh,v 1.2 2005/03/10 21:35:03 aivazis Exp $

# End of file
