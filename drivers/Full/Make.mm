# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE = Full

PROJECT = CitcomS
PACKAGE = drivers/$(TYPE)

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_BIN = $(BLD_BINDIR)/$(PROJECT)$(TYPE)
PROJ_LIBS = $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE).$(EXT_LIB)

#PROJ_CC_INCLUDES = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_CC_INCLUDES = ../../lib/Common

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
# $Id: Make.mm,v 1.4 2003/08/07 18:41:42 tan2 Exp $

#
# End of file
