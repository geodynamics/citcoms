#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Stokes_solver import Stokes_solver
import CitcomS.Regional as Regional


class ImcompressibleNewtonian(Stokes_solver):


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




class ImcompressibleNonNewtonian(Stokes_solver):

    # over-ride Stokes.solver.run()
    def run(self, *args, **kwds):

	self.init()

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

	self.fini()

	return


    def init(self):
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
__id__ = "$Id: Imcompressible.py,v 1.4 2003/06/23 20:54:13 tan2 Exp $"

# End of file
