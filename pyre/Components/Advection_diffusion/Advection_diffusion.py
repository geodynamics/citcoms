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

from mpi.Application import Application
import CitcomS.Regional as Regional
    
class PG_timestep(Application):

    def __init__(self,field_name='temp'):
        Application.__init__(self, "PG_timestep")
        if(field_name=='temp'):
            self.Init=Regional.PG_timestep_init
            self.Solve=Regional.PG_timestep_solve
            self.Control_timemarching=Regional.PG_timemarching_control
            self.Fini=Regional.PG_timestep_fini
##      if(self.field_name=="comp"):
##            self.Init=Regional.PG_timestep_init_comp
##            self.Solve=Regional.PG_timestep_solve_comp
##            self.Control_timemarching=Regional.PG_timemarching_control_comp
##            self.Fini=Regional.PG_timestep_fini_comp
        return
        
    def main(self):
	raise NotImplementedError, \
	      "PG Solver doesn't run stand-along. Call init(), run(), fini() sequentially in stead."
	return
    
    def run(self):
        self.Solve()
        self.Control_timemarching()
        return

    def init(self):
        Application.init(self)
        self.Init()
        return

    def fini(self):
        self.Fini()      
        Application.fini(self)
        return
    
    class Facilities(Application.Facilities):

        __facilities__ = Application.Facilities.__facilities__ + (
            )


    class Properties(Application.Properties):


        __properties__ = Application.Properties.__properties__ + (
            )


# version
__id__ = "$Id: Advection_diffusion.py,v 1.1 2003/05/22 18:20:21 ces74 Exp $"

# End of file 
