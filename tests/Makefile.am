## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
## Copyright (C) 2002-2005, California Institute of Technology.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##</LicenseText>

# $Id$

nodist_bin_SCRIPTS = \
	citcomsfull.sh \
	citcomsregional.sh
do_subst = sed \
	-e 's|[@]pkgpythondir[@]|$(pkgpythondir)|g' \
	-e 's|[@]PYTHON[@]|$(PYTHON)|g' \
	-e 's|[@]PYTHONPATH[@]|$(PYTHONPATH)|g'
citcomsfull.sh: $(srcdir)/citcomsfull.sh.in
	$(do_subst) < $(srcdir)/citcomsfull.sh.in > $@ || (rm -f $@ && exit 1)
citcomsregional.sh: $(srcdir)/citcomsregional.sh.in
	$(do_subst) < $(srcdir)/citcomsregional.sh.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_bin_SCRIPTS)
EXTRA_DIST = \
	citcomsfull.sh.in \
	citcomsregional.sh.in

EXTRA_DIST += \
	asap.py \
	hrothgar.py \
	signon.py

## end of Makefile.am
