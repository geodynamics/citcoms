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
            self.init=Regional.PG_timestep_init
            self.solve=Regional.PG_timestep_solve
##      if(self.field_name=="comp"):
##            self.solve=Regional.PG_timestep_solve_comp
##            self.init=Regional.PG_timestep_init_comp
        return
        
    def main(self):
	raise NotImplementedError, \
	      "PG Solver doesn't run stand-along. Call init(), run(), fini() sequentially in stead."
	return
    
    def run(self):
        self.solve()
        return

    def init(self):
        Application.init(self)
        self.init()
        return

    class Facilities(Application.Facilities):

        __facilities__ = Application.Facilities.__facilities__ + (
            )


    class Properties(Application.Properties):


        __properties__ = Application.Properties.__properties__ + (
            )


# version
__id__ = "$Id: Advection_diffusion.py,v 1.3 2003/05/23 04:22:04 ces74 Exp $"

# End of file 
