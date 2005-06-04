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


    def __init__(self, name, facility):
        Sphere.__init__(self, name, facility)
	self.inventory.nproc_surf = 1
        return



    def launch(self):
        self.CitcomModule.regional_sphere_launch(self.all_variables)
	return



    class Inventory(Sphere.Inventory):

        import pyre.inventory


        # used only in Regional version, not in Full version
        theta_min = pyre.inventory.float("theta_min", default=1.0708)
        theta_max = pyre.inventory.float("theta_max", default=2.0708)
        fi_min = pyre.inventory.float("fi_min", default=0.0)
        fi_max = pyre.inventory.float("fi_max", default=1.0)



# version
__id__ = "$Id: RegionalSphere.py,v 1.12 2005/06/03 21:51:45 leif Exp $"

# End of file
