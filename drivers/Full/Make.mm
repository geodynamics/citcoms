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
PROJ_LIBS = -l$(PROJECT)Common -l$(PROJECT)$(TYPE) # NOTE: the order seems to be important.

#PROJ_CC_INCLUDES = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_CC_INCLUDES = ../../lib/Common

PROJ_LCC_FLAGS = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm

PROJ_SRCS = \
	../Citcom.c

PROJ_OBJS = ${addprefix ${PROJ_TMPDIR}/, ${addsuffix .${EXT_OBJ}, ${basename ${notdir ${PROJ_SRCS}}}}}

PROJ_CLEAN += $(PROJ_BIN) $(PROJ_OBJS)


all: $(PROJ_BIN)

.PHONY: $(PROJ_BIN)
$(PROJ_BIN):
	$(CC) $(PROJ_CC_FLAGS) -o $@ $(PROJ_OBJS) $(PROJ_LIBS) $(LCFLAGS)

$(PROJ_OBJS): $(PROJ_SRCS)
	$(CC_COMPILE_COMMAND) $(PROJ_CC_FLAGS)



# version
# $Id: Make.mm,v 1.7 2003/11/26 23:27:09 tan2 Exp $

#
# End of file
