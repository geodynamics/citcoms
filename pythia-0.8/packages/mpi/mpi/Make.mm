# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = mpi

#--------------------------------------------------------------------------
#

all: export

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    Application.py \
    CartesianCommunicator.py \
    Communicator.py \
    CommunicatorGroup.py \
    DummyCommunicator.py \
    Launcher.py \
    LauncherMPICH.py \
    LauncherPBS.py \
    Port.py \
    Timer.py \
    TimingCenter.py \
    __init__.py


export:: export-python-modules

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

# End of file
