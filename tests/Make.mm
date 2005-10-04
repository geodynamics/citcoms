# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include Python/default.def

PROJECT = CitcomS
PACKAGE = tests

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

PROJ_CXX_INCLUDES = ../../Exchanger/lib
PROJ_BIN = $(BLD_BINDIR)/array2d
PROJ_CLEAN += $(PROJ_BIN)

PROJ_PYTHONTESTS = signon.py
PROJ_CPPTESTS = $(PROJ_BIN)
PROJ_EMPTYTESTS = $(BLD_BINDIR)/CitcomSFull $(BLD_BINDIR)/CitcomSRegional
PROJ_TESTS = $(PROJ_PYTHONTESTS) $(PROJ_CPPTESTS) $(PROJ_EMPTYTESTS)

#--------------------------------------------------------------------------
#

all: $(PROJ_TESTS) export

test:
	for test in $(PROJ_TESTS) ; do $${test}; done; exit 0

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#

EXPORT_BINS = \
    array2d \
    citcomsfull.sh \
    citcomsregional.sh \
    coupledcitcoms.sh

RELEASE_BINARIES = $(foreach bin,$(EXPORT_BINS),$(PROJ_BINDIR)/$(bin))

export:: release-binaries

release-binaries:: $(RELEASE_BINARIES)

$(PROJ_BINDIR)/%.sh: %.sh.in
	sed \
		-e 's|[@]pkgpythondir[@]|$(EXPORT_MODULEDIR)|g' \
		-e 's|[@]PYTHON[@]|$(PYTHON)|g' \
		-e 's|[@]PYTHONPATH[@]|$(EXPORT_ROOT)/modules|g' \
		$< > $@ || (rm -f $@ && exit 1)
	$(CHMOD) +x $@

#--------------------------------------------------------------------------
#

$(PROJ_BIN): array2d.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LCXXFLAGS) \
		-lExchanger -l_mpimodule -ljournal \
		$(PYTHON_APILIB) $(EXTERNAL_LIBS)

# version
# $Id$

# End of file
