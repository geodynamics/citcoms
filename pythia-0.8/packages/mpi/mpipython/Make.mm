# -*- Makefile -*-
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

PROJECT = mpipython

PROJ_BIN = $(BLD_BINDIR)/$(PROJECT).exe
MPIPYTHON = $(PROJECT).$(EXT_CXX)

EXPORT_BINS = $(PROJ_BIN)
PROJ_BINDIR = $(EXPORT_BINDIR)

LIBRARIES = $(PYTHON_APILIB) $(EXTERNAL_LIBS) $(LCXX_FORTRAN)

PROJ_CLEAN += $(PROJ_BIN)

all: $(PROJ_BIN) export

export:: export-binaries

install: $(PROJ_BIN)
	$(CP_F) $(PROJ_BIN) $(TOOLS_DIR)/bin

$(PROJ_BIN): $(MPIPYTHON)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LCXXFLAGS) $(LIBRARIES)


# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

# End of file
