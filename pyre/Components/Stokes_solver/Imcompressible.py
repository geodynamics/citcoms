#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Stokes_solver import Stokes_solver
import CitcomS.Regional as Regional


class ImcompressibleNewtionian(Stokes_solver):

    def run(self, *args, **kwds):

	self.conform_vbcs()
	self.get_forces()
	self.get_viscosity()
	self.get_stiffness()
	self.solve()

	return


    def preInit(self):
	# read/check input parameters here
	return


    def postInit(self):
	# allocate and initialize memory here

	# assign function objects
	self.get_viscosity = Regional.get_system_viscosity
	self.conform_vbcs = Regional.velocities_conform_bcs
	self.get_forces = Regional.assemble_forces
	self.get_stiffness = Regional.construct_stiffness_B_matrix
	self.solve = Regional.solve_constrained_flow_iterative
	return


    def fini(self):
	# free memory here
	return





class ImcompressibleNonNewtionian(Stokes_solver):

    def run(self, *args, **kwds):
	Regional.velocities_conform_bcs()
	Regional.assemble_forces()
	
	while (self.count < 50) and self.sdepv_not_convergent:
	    if Control.viscosity.updated_allowed:
		Regional.get_system_viscosity()

	    Regional.construct_stiffness_B_matrix()
	    self.viscosity_misfit = Regional.solve_constrained_flow_iterative()

	    if Control.viscosity.SDEPV:
		Regional.general_stokes_solver_update_velo()
		self.Udot_mag = Regional.general_stokes_solver_Unorm()
		self.dUdot_mag = Regional.general_stokes_solver_Udotnorm()

		self.sdepv_not_convergent = self.sdepv_not_convergent or (self.dUdot_mag > self.viscosity_misfit)

		Regional.general_stokes_solver_log(self.Udot_mag, self.dUdot_mag, self.count)

	    self.count += 1
	return


    def preInit(self):
	self.time = Regional.CPU_time()
	# read/check input parameters here
	return


    def postInit(self):
	# allocate and initialize memory here
	Regional.general_stokes_solver_init()

	self.Udot_mag = 0
	self.dUdot_mag = 0
	self.count = 1
	self.sdepv_not_convergent = Control.viscosity.SDEPV
	return


    def fini(self):
	# free memory here
	Regional.general_stokes_solver_fini()
	return








# version
__id__ = "$Id: Imcompressible.py,v 1.2 2003/05/16 21:11:54 tan2 Exp $"

# End of file 
