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
	mayavi2_citcoms_display.py \
	autocombine.py \
	batchcombine.py \
	batchsurf.py \
	citcoms_datadir \
	combine.py \
	combinesurf.py \
	dxgeneral.py \
	dxgeneralsurf.sh \
	estimate_size.py \
	getcoord.sh \
	getlog.sh \
	getsurf.sh \
	h5tocap.py \
	miff2avi \
	miff2mpg \
	parser.py \
	pasteCitcomData.py \
	plot_annulus.py \
	plot_layer.py \
	zslice.py

nobase_dist_visual_DATA = \
	Mayavi2/citcoms_plugins/__init__.py \
	Mayavi2/citcoms_plugins/plugins/__init__.py \
	Mayavi2/citcoms_plugins/plugins/CitcomS_hdf_file_reader.py \
	Mayavi2/citcoms_plugins/plugins/CitcomSHDFUgrid.py \
	Mayavi2/citcoms_plugins/plugins/OpenCitcomSFILES.py \
	Mayavi2/citcoms_plugins/plugins/filter/CitcomSFilterActions.py \
	Mayavi2/citcoms_plugins/plugins/filter/__init__.py \
	Mayavi2/citcoms_plugins/plugins/filter/CitcomSshowSurface.py \
	Mayavi2/citcoms_plugins/plugins/filter/CitcomSSphere.py \
	Mayavi2/citcoms_plugins/plugins/filter/CitcomsCapFilter.py \
	Mayavi2/citcoms_plugins/plugins/filter/CitcomsReduceFilter.py \
	Mayavi2/citcoms_plugins/utils.py \
	Mayavi2/citcoms_plugins/custom_ui.py \
	Mayavi2/citcoms_plugins/README \
	Mayavi2/README \
	Mayavi2/mayavi_custom_ui.py \
	samples/cookbook1.h5 \
	samples/cookbook1.100.h5 \
	samples/fulltest.cap00.100 \
	samples/fulltest.cap00.100.general \
	samples/fulltest.cap01.100 \
	samples/fulltest.cap01.100.general \
	samples/fulltest.cap02.100 \
	samples/fulltest.cap02.100.general \
	samples/fulltest.cap03.100 \
	samples/fulltest.cap03.100.general \
	samples/fulltest.cap04.100 \
	samples/fulltest.cap04.100.general \
	samples/fulltest.cap05.100 \
	samples/fulltest.cap05.100.general \
	samples/fulltest.cap06.100 \
	samples/fulltest.cap06.100.general \
	samples/fulltest.cap07.100 \
	samples/fulltest.cap07.100.general \
	samples/fulltest.cap08.100 \
	samples/fulltest.cap08.100.general \
	samples/fulltest.cap09.100 \
	samples/fulltest.cap09.100.general \
	samples/fulltest.cap10.100 \
	samples/fulltest.cap10.100.general \
	samples/fulltest.cap11.100 \
	samples/fulltest.cap11.100.general \
	samples/regtest.cap00.100 \
	samples/regtest.cap00.100.general \
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
	  if test -f "$$p.in"; then d=; else d="$(srcdir)/"; fi; \
	  f=$(am__strip_dir) \
	  echo " $(do_install) '$$d$$p.in' > '$(DESTDIR)$(visualdir)/$$f'"; \
	  $(do_install) "$$d$$p.in" > "$(DESTDIR)$(visualdir)/$$f"; \
	  echo " chmod 644 '$(DESTDIR)$(visualdir)/$$f'"; \
	  chmod 644 "$(DESTDIR)$(visualdir)/$$f"; \
	done

if COND_HDF5
    bin_PROGRAMS = h5tocap h5tovelo project_geoid
    h5tocap_SOURCES = h5tocap.c h5util.c h5util.h
    h5tovelo_SOURCES = h5tovelo.c h5util.c h5util.h
    h5tovelo_LDADD = $(LIBHDF5)
else
    bin_PROGRAMS = project_geoid
endif

project_geoid_SOURCES = project_geoid.c

CLEANFILES = $(nodist_visual_DATA)
EXTRA_DIST = $(visual_templates)

## end of Makefile.am
