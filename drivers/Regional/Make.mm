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
# $Id: Make.mm,v 1.2 2003/04/03 19:38:36 tan2 Exp $

include local.def
TYPE = Regional

PROJECT = CitcomS/drivers/$(TYPE)
PACKAGE = CitcomS$(TYPE)

PROJ_BIN = $(BLD_BINDIR)/$(PACKAGE)
PROJ_LIBS = $(BLD_LIBDIR)/lib$(PACKAGE).$(EXT_LIB) $(BLD_LIBDIR)/lib$(PACKAGE)Common.$(EXT_LIB)

#PROJ_CC_INCLUDES = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_CC_INCLUDES = ../../lib/Common ../../lib/$(TYPE)

PROJ_LCC_FLAGS = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm

PROJ_SRCS = \
	../Citcom.c

PROJ_OBJS = ${addprefix ${PROJ_TMPDIR}/, ${addsuffix .${EXT_OBJ}, ${basename ${notdir ${PROJ_SRCS}}}}}

PROJ_CLEAN += $(PROJ_BIN) $(PROJ_OBJS)


all: $(PROJ_BIN)

$(PROJ_BIN): $(PROJ_OBJS) $(PROJ_LIBS)
	$(CC) $(PROJ_CC_FLAGS) -o $@ $^ $(LCFLAGS)

$(PROJ_OBJS): $(PROJ_SRCS)
	$(CC_COMPILE_COMMAND) $(PROJ_CC_FLAGS) 



# version
# $Id: Make.mm,v 1.2 2003/04/03 19:38:36 tan2 Exp $

#
# End of file
