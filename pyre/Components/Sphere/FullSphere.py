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
        self.CitcomModule.full_sphere_launch(self.all_variables)
	return




# version
__id__ = "$Id: FullSphere.py,v 1.2 2003/08/19 21:24:35 tan2 Exp $"

# End of file
