#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Stokes_solver import Stokes_solver


class ImcompressibleNewtonian(Stokes_solver):


    def _form_RHS(self):
	self.CitcomModule.velocities_conform_bcs()
	self.CitcomModule.assemble_forces()
	return


    def _form_LHS(self):
	self.CitcomModule.get_system_viscosity()
	self.CitcomModule.construct_stiffness_B_matrix()
	return


    def _solve(self):
	misfit = self.CitcomModule.solve_constrained_flow_iterative()
        return misfit



class ImcompressibleNonNewtonian(ImcompressibleNewtonian):

    # over-ride Stokes_solver.run()
    def run(self):

	self._myinit()

	self._form_RHS()

	while (self.count < 50) and self.sdepv_not_convergent:

	    self._form_LHS()
	    self.viscosity_misfit = self._solve()

	    self.CitcomModule.general_stokes_solver_update_velo()
	    self.Udot_mag = self.CitcomModule.general_stokes_solver_Unorm()
	    self.dUdot_mag = self.CitcomModule.general_stokes_solver_Udotnorm()

	    self.CitcomModule.general_stokes_solver_log(self.Udot_mag, self.dUdot_mag,
					       self.count)

	    self.sdepv_not_convergent = (self.dUdot_mag > self.viscosity_misfit)
	    self.count += 1

	self._myfini()

	return


    def _myinit(self):
	# allocate and initialize memory here
	self.CitcomModule.general_stokes_solver_init()

	self.Udot_mag = 0
	self.dUdot_mag = 0
	self.count = 1
	self.sdepv_not_convergent = True
	return


    def _myfini(self):
	# free memory here
	self.CitcomModule.general_stokes_solver_fini()
	return








# version
__id__ = "$Id: Imcompressible.py,v 1.7 2003/07/24 17:46:47 tan2 Exp $"

# End of file
