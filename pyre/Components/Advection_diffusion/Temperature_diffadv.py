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

from Advection_diffusion import Advection_diffusion
import CitcomS.Regional as Regional
    
class Temperature_diffadv(Advection_diffusion):

    def _init(self):
        Regional.PG_timestep_init()
        return

    def _solve(self):
        Regional.PG_timestep_solve

    def main(self):
	raise NotImplementedError, \
	      "PG Solver doesn't run stand-along. Call init(), run(), fini() sequentially in stead."
	return
    
# version
__id__ = "$Id: Temperature_diffadv.py,v 1.1 2003/07/03 23:43:20 ces74 Exp $"

# End of file 
