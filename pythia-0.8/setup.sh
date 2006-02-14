#!/bin/sh
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

  export TOOLS_DIR=/home/tools
  export TOOLS_INCDIR=${TOOLS_DIR}/include
  export TOOLS_LIBDIR=${TOOLS_DIR}/lib
  export LD_LIBRARY_PATH=${TOOLS_DIR}/lib
  export PATH=${TOOLS_DIR}/bin:${PATH}

# local geometry

  export PYTHIA_VERSION=0.8

  export DV_DIR=${HOME}/dv
  export PYTHIA_DIR=${DV_DIR}/pythia-${PYTHIA_VERSION}
  export TEMPLATES_DIR=${DV_DIR}/templates

  export BLD_ROOT=${DV_DIR}/builds/pythia-${PYTHIA_VERSION}
  export EXPORT_ROOT=${TOOLS_DIR}/pythia-${PYTHIA_VERSION}

  export PATH=${PATH}:${EXPORT_ROOT}/bin
  export PYTHONPATH=${EXPORT_ROOT}/modules
  export LD_LIBRARY_PATH=${EXPORT_ROOT}/lib:${LD_LIBRARY_PATH}

# build system

  export GNU_MAKE=make
  export DEVELOPER=aivazis
  export TARGET=shared,mpi,debug
  export TARGET_F77=Absoft-8.0

  export BLD_CONFIG=${DV_DIR}/config
  export PATH=${BLD_CONFIG}/make:${PATH}

# Python

  export PYTHON_VERSION=2.3
  export PYTHON_DIR=/usr
  export PYTHON_LIBDIR=${PYTHON_DIR}/lib/python${PYTHON_VERSION}
  export PYTHON_INCDIR=${PYTHON_DIR}/include/python${PYTHON_VERSION}
  export PYTHONSTARTUP=${HOME}/.python
  export PATH=${PYTHON_DIR}/bin:${PATH}

# CVS

  export CVS_RSH=ssh

# mpi

  export MPI_VERSION=1.2.5
  export MPI_DIR=${TOOLS_DIR}/mpich-${MPI_VERSION}_absoft8
  export MPI_INCDIR=${MPI_DIR}/include
  export MPI_LIBDIR=${MPI_DIR}/lib
  export PATH=${MPI_DIR}/bin:${PATH}

# version
# $Id: setup.sh,v 1.2 2005/03/10 21:35:09 aivazis Exp $

# End of file 
