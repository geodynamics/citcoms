# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                              CitcomS
#                 ---------------------------------
#
#                              Authors:
#           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
#           Clint Conrad, Michael Gurnis, and Eun-seo Choi
#          (c) California Institute of Technology 1994-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
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

EXPORT_BINS = $(PROJECT)$(TYPE)

all: $(PROJ_BIN) release-binaries

.PHONY: $(PROJ_BIN)
$(PROJ_BIN): $(PROJ_OBJS)
	$(CC) $(PROJ_CC_FLAGS) -o $@ $(PROJ_OBJS) $(PROJ_LIBS) $(LCFLAGS)

$(PROJ_OBJS): $(PROJ_SRCS)
	$(CC_COMPILE_COMMAND) $(PROJ_CC_FLAGS)



# version
# $Id: Make.mm,v 1.11 2005/06/10 02:23:11 leif Exp $

#
# End of file
