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


class Imcompressible(Stokes_solver):

    def run(self, *args, **kwds):
	time = Regional.CPU_time()
	Regional.velocities_conform_bcs()
	Regional.assemble_forces(0)
	
	Udot_mag = 0
	dUdot_mag = 0
	count = 1
	sdepv_not_convergent = Control.viscosity.SDEPV

	while (count < 50) and sdepv_not_convergent:
	    if Control.viscosity.updated_allowed:
		Regional.get_system_viscosity()

	    Regional.construct_stiffness_B_matrix()
	    viscosity_misfit = Regional.solve_constrained_flow_interative()

	    if Control.viscosity.SDEPV:
		Regional.general_stokes_solver_update_velo()
		Udot_mag = Regional.general_stokes_solver_Unorm()
		dUdot_mag = Regional.general_stokes_solver_Udotnorm()

		sdepv_not_convergent = sdepv_not_convergent or (dUdot_mag > viscosity_misfit)

		Regional.general_stokes_solver_log(Udot_mag, dUdot_mag)

	return


    def preInit(self):
	# read/check input parameters here
	return


    def postInit(self):
	# allocate and initialize memory here
	Regional.general_stokes_solver_init()
	return


    def fini(self):
	# free memory here
	Regional.general_stokes_solver_fini()
	return
    

    def __init__(self,name):
	Stokes_solver.__init__(name)
	return







# version
__id__ = "$Id: Imcompressible.py,v 1.1.1.1 2003/05/15 00:07:54 tan2 Exp $"

# End of file 
