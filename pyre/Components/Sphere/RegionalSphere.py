#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomS.Components.CitcomComponent import CitcomComponent

class RegionalSphere(CitcomComponent):


    def __init__(self, name, facility, CitcomModule):
        # bind component method to facility method
        CitcomModule.mesher_set_properties = CitcomModule.RegionalSphere_set_properties

        CitcomComponent.__init__(self, name, facility, CitcomModule)
        return




    def run(self):
        start_time = self.CitcomModule.CPU_time()
        self.CitcomModule.regional_sphere_setup()

        import mpi
        if not mpi.world().rank:
            print "initialization time = %f" % \
                  (self.CitcomModule.CPU_time() - start_time)

	return



    def init(self, parent):
        self.CitcomModule.regional_sphere_init()
        return



    def fini(self):
	return



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties
        from math import pi

        inventory = [

            pyre.properties.bool("coor", False),
            pyre.properties.str("coor_file", "coor.dat"),

            pyre.properties.int("nodex", 9),
            pyre.properties.int("nodey", 9),
            pyre.properties.int("nodez", 9),
            pyre.properties.int("mgunitx", 8),
            pyre.properties.int("mgunity", 8),
            pyre.properties.int("mgunitz", 8),
            pyre.properties.int("levels", 1),

            pyre.properties.float("radius_outer", 1.0),
            pyre.properties.float("radius_inner", 0.55),

            pyre.properties.float("theta_min", pi/2),
            pyre.properties.float("theta_max", pi*3/4),
            pyre.properties.float("fi_min", pi*3/4),
            pyre.properties.float("fi_max", pi*5/4),

            ]


# version
__id__ = "$Id: RegionalSphere.py,v 1.4 2003/07/24 20:10:33 tan2 Exp $"

# End of file
