#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Sphere import Sphere

class RegionalSphere(Sphere):


    def __init__(self, name, facility, CitcomModule):
        Sphere.__init__(self, name, facility, CitcomModule)
	self.inventory.nproc_surf = 1
        return



    def launch(self):
        self.CitcomModule.regional_sphere_launch(self.all_variables)
	return




# version
__id__ = "$Id: RegionalSphere.py,v 1.8 2003/08/19 21:24:35 tan2 Exp $"

# End of file
