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

    def form_RHS(self):
	Regional.velocities_conform_bcs()
	Regional.assemble_forces()
	return


    def form_LHS(self):
	Regional.get_system_viscosity()
	Regional.construct_stiffness_B_matrix()
	return


    def solve(self):
	return Regional.solve_constrained_flow_iterative()
    

    def preInit(self):
	# read/check input parameters here
	return


    def postInit(self):
	# allocate and initialize memory here
	return


    def fini(self):
	# free memory here
	return





class ImcompressibleNonNewtionian(ImcompressibleNewtionian):

    def run(self, *args, **kwds):

	self.form_RHS()
	
	while (self.count < 50) and self.sdepv_not_convergent:

	    self.form_LHS()
	    self.viscosity_misfit = self.solve()

	    Regional.general_stokes_solver_update_velo()
	    self.Udot_mag = Regional.general_stokes_solver_Unorm()
	    self.dUdot_mag = Regional.general_stokes_solver_Udotnorm()

	    Regional.general_stokes_solver_log(self.Udot_mag, self.dUdot_mag,
					       self.count)

	    self.sdepv_not_convergent = (self.dUdot_mag > self.viscosity_misfit)
	    self.count += 1
	    
	return


    def preInit(self):
	# read/check input parameters here
	return


    def postInit(self):
	# allocate and initialize memory here
	Regional.general_stokes_solver_init()

	self.Udot_mag = 0
	self.dUdot_mag = 0
	self.count = 1
	self.sdepv_not_convergent = True
	return


    def fini(self):
	# free memory here
	Regional.general_stokes_solver_fini()
	return








# version
__id__ = "$Id: Incompressible.py,v 1.3 2003/05/20 18:56:58 tan2 Exp $"

# End of file 
