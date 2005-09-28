## Process this file with automake to produce Makefile.in
# $Id$

nobase_pkgpython_PYTHON = \
	Components/Advection_diffusion/Advection_diffusion.py \
	Components/Advection_diffusion/__init__.py \
	Components/BC.py \
	Components/CitcomComponent.py \
	Components/Const.py \
	Components/Exchanger/CoarseGridExchanger.py \
	Components/Exchanger/Exchanger.py \
	Components/Exchanger/FineGridExchanger.py \
	Components/Exchanger/__init__.py \
	Components/Exchanger/Inlet.py \
	Components/Exchanger/Outlet.py \
	Components/IC.py \
	Components/__init__.py \
	Components/Launchers.py \
	Components/Param.py \
	Components/Phase.py \
	Components/Sphere/FullSphere.py \
	Components/Sphere/__init__.py \
	Components/Sphere/RegionalSphere.py \
	Components/Sphere/Sphere.py \
	Components/Stokes_solver/Incompressible.py \
	Components/Stokes_solver/__init__.py \
	Components/Tracer.py \
	Components/Visc.py \
	Controller.py \
	CoupledApp.py \
	Coupler.py \
	Layout.py \
	SimpleApp.py \
	Solver/FullSolver.py \
	Solver/__init__.py \
	Solver/RegionalSolver.py \
	Solver/Solver.py

nodist_pkgpython_PYTHON = \
	config.py \
	__init__.py
do_subst = sed \
	-e 's|[@]PACKAGE_VERSION[@]|$(PACKAGE_VERSION)|g'
config.py: $(top_builddir)/portinfo $(top_builddir)/Makefile
	$(PYCONFIG) -h $(top_builddir)/portinfo -m $(top_builddir)/Makefile -o $@
__init__.py: $(srcdir)/__init__.py.in
	$(do_subst) < $(srcdir)/__init__.py.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_pkgpython_PYTHON)
EXTRA_DIST = \
	__init__.py.in

## end of Makefile.am
