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

        import pyre.properties

        inventory = [

            # used only in Regional version, not in Full version
            pyre.properties.float("theta_min", 1.0708),
            pyre.properties.float("theta_max", 2.0708),
            pyre.properties.float("fi_min", 0.0),
            pyre.properties.float("fi_max", 1.0),

            ]


# version
__id__ = "$Id: RegionalSphere.py,v 1.11 2004/06/24 19:25:48 tan2 Exp $"

# End of file
