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


    def __init__(self, name, facility):
        Sphere.__init__(self, name, facility)
	self.inventory.nproc_surf = 12
        return



    def launch(self):
        self.CitcomModule.full_sphere_launch(self.all_variables)
	return




# version
__id__ = "$Id: FullSphere.py,v 1.3 2003/08/27 20:52:47 tan2 Exp $"

# End of file
