# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include Python/default.def

PROJECT = CitcomS
PACKAGE = tests

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

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
    citcomsfull.sh \
    citcomsregional.sh

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

# version
# $Id$

# End of file
