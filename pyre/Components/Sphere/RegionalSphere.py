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
            pyre.properties.float("theta_min", 1.5),
            pyre.properties.float("theta_max", 1.8),
            pyre.properties.float("fi_min", 0.0),
            pyre.properties.float("fi_max", 0.4),

            ]


# version
__id__ = "$Id: RegionalSphere.py,v 1.10 2003/10/28 23:51:48 tan2 Exp $"

# End of file
