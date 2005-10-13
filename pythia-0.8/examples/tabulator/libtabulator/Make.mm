# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

PROJECT = tabulator
PACKAGE = libtabulator

PROJ_AR = $(BLD_LIBDIR)/$(PACKAGE).$(EXT_AR)
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_TIDY += *.mod
PROJ_CLEAN += $(PROJ_INCDIR) *.mod
PROJ_F90_MODULES = $(PROJ_TMPDIR)

PROJ_SRCS = \
    tabulator.f90 \
    exponential.f90 \
    quadratic.f90 \
    simpletab.f90


all: $(PROJ_AR) export

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build the shared object

$(PROJ_AR): product_dirs $(PROJ_OBJS)
	$(AR) $(AR_CREATE_FLAGS) $(PROJ_AR) $(PROJ_OBJS)
	$(RANLIB) $(PROJ_AR)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# export

export:: export-libraries

EXPORT_LIBS = $(PROJ_SAR)

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

# End of file
