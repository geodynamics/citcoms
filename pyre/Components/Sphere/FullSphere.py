#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Sphere import Sphere

class FullSphere(Sphere):


    def __init__(self, name, facility, CitcomModule):
        Sphere.__init__(self, name, facility, CitcomModule)
	self.inventory.nproc_surf = 12
        return



    def launch(self):
        self.CitcomModule.full_sphere_launch()
	return




# version
__id__ = "$Id: FullSphere.py,v 1.1 2003/08/01 19:05:36 tan2 Exp $"

# End of file
