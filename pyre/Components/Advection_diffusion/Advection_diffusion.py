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

from pyre.components.Component import Component

class Advection_diffusion(Component):

    def __init__(self,name,facility="tsolver"):
        Component.__init__(self, name, facility)
        self._setProperties()
        return

    def _setProperties(self):
        import CitcomS.Regional as Regional
        #Regional.general_stokes_solver_set_prop(self.properties)
        return

    def run(self):
        #test
        print self.properties.ADV
        return
    
    def _init(self):
	return


    def _fini(self):
	return

    def _solve(self):
	return


    def _output(self, *args, **kwds):
	return


    class Properties(Component.Properties):

        import pyre.properties as prop

        __properties__ = (
            
            prop.bool("ADV",True),
            prop.float("fixed_timestep",0.0),
            prop.float("finetunedt",0.7),

            prop.int("adv_sub_iterations",2),
            prop.float("maxadvtime",10.0),
            prop.bool("precond",True),

            prop.bool("aug_lagr",True),
            prop.float("aug_number",2.0e3),

	    )


# version
__id__ = "$Id: Advection_diffusion.py,v 1.4 2003/07/03 23:43:20 ces74 Exp $"

# End of file 
