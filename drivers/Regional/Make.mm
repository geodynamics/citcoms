# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2003  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# version
# $Id: Make.mm,v 1.1 2003/03/24 01:46:37 tan2 Exp $

include local.def
TYPE = Regional

PROJECT = CitcomS
PROJ_BIN = $(BLD_BINDIR)/$(PROJECT)$(TYPE)
PACKAGE = CitcomS$(TYPE)
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(TYPE)

PROJ_LIBS = $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE).$(EXT_LIB) $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE)Common.$(EXT_LIB)

PROJ_CC_INCLUDES = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_LCC_FLAGS = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm

PROJ_SRCS = \
	Citcom.c

PROJ_CLEAN += $(PROJ_BIN)


all: $(PROJ_BIN)

$(PROJ_BIN): $(PROJ_OBJS) $(PROJ_LIBS)
	$(CC) $(PROJ_CC_FLAGS) -o $@ $^ $(LCFLAGS)

# version
# $Id: Make.mm,v 1.1 2003/03/24 01:46:37 tan2 Exp $

#
# End of file
