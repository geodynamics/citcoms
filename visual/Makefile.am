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

SUBDIRS = OpenDXMacro

SUFFIXES = .net.in .net

visualdir = $(pkgdatadir)/visual
dxmacrodir = $(visualdir)/OpenDXMacro
abs_builddir = @abs_builddir@

dist_bin_SCRIPTS = \
	autocombine.py \
	batchcombine.py \
	batchpaste.sh \
	batchsurf.py \
	combine.py \
	combinesurf.py \
	dxgeneral.sh \
	dxgeneralsurf.sh \
	getcoord.sh \
	getlog.sh \
	getsurf.sh \
	h52combined.py \
	miff2avi \
	miff2mpg \
	parser.py \
	pasteCitcomData.sh \
	zslice.py

nobase_dist_visual_DATA = \
	samples/fulltest.cap0.0 \
	samples/fulltest.cap0.0.general \
	samples/fulltest.cap1.0 \
	samples/fulltest.cap10.0 \
	samples/fulltest.cap10.0.general \
	samples/fulltest.cap1.0.general \
	samples/fulltest.cap11.0 \
	samples/fulltest.cap11.0.general \
	samples/fulltest.cap2.0 \
	samples/fulltest.cap2.0.general \
	samples/fulltest.cap3.0 \
	samples/fulltest.cap3.0.general \
	samples/fulltest.cap4.0 \
	samples/fulltest.cap4.0.general \
	samples/fulltest.cap5.0 \
	samples/fulltest.cap5.0.general \
	samples/fulltest.cap6.0 \
	samples/fulltest.cap6.0.general \
	samples/fulltest.cap7.0 \
	samples/fulltest.cap7.0.general \
	samples/fulltest.cap8.0 \
	samples/fulltest.cap8.0.general \
	samples/fulltest.cap9.0 \
	samples/fulltest.cap9.0.general \
	samples/regtest.cap0.0 \
	samples/regtest.cap0.0.general \
	visFull.cfg \
	visRegional.cfg \
	visRegionalSurface.cfg

nodist_visual_DATA = \
	visFull.net \
	visRegional.net \
	visRegionalSurface.net

visual_templates = \
	visFull.net.in \
	visRegional.net.in \
	visRegionalSurface.net.in

do_build = sed -e s:[@]dxmacrodir[@]:$(abs_builddir)/OpenDXMacro:g
do_install = sed -e s:[@]dxmacrodir[@]:$(dxmacrodir):g

.net.in.net:
	$(do_build) $< > $@ || (rm -f $@ && exit 1)

install-nodist_visualDATA: $(nodist_visual_DATA)
	@$(NORMAL_INSTALL)
	test -z "$(visualdir)" || $(mkdir_p) "$(DESTDIR)$(visualdir)"
	@list='$(nodist_visual_DATA)'; for p in $$list; do \
	  if test -f "$$p"; then d=; else d="$(srcdir)/"; fi; \
	  f=$(am__strip_dir) \
	  echo " $(do_install) '$$d$$p.in' > '$(DESTDIR)$(visualdir)/$$f'"; \
	  $(do_install) "$$d$$p.in" > "$(DESTDIR)$(visualdir)/$$f"; \
	  echo " chmod 644 '$(DESTDIR)$(visualdir)/$$f'"; \
	  chmod 644 "$(DESTDIR)$(visualdir)/$$f"; \
	done

CLEANFILES = $(nodist_visual_DATA)
EXTRA_DIST = $(visual_templates)

## end of Makefile.am
