#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component
import CitcomS.Regional as Regional


class RegionalSphere(Component):


    def __init__(self, name, facility="mesher"):
        Component.__init__(self, name, facility)
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
        Regional.RegionalSphere_set_properties(self.inventory)
        return



    def run(self):
        start_time = Regional.CPU_time()
        #Regional.mesher_setup()

        import mpi
        if not mpi.world().rank:
            print "initialization time = %f" % \
                  (Regional.CPU_time() - start_time)

	return



    def init(self, parent):
        Regional.set_3dsphere_defaults()
        return



    #def fini(self):
	#return



    class Inventory(Component.Inventory):

        import pyre.properties
        from math import pi

        inventory = [

            pyre.properties.bool("coor", False),
            pyre.properties.str("coor_file", "coor.dat"),

            pyre.properties.int("nodex", 17),
            pyre.properties.int("nodey", 17),
            pyre.properties.int("nodez", 9),
            pyre.properties.int("mgunitx", 8),
            pyre.properties.int("mgunity", 8),
            pyre.properties.int("mgunitz", 8),
            pyre.properties.int("levels", 3),

            pyre.properties.float("radius_outer", 1.0),
            pyre.properties.float("radius_inner", 0.55),

            pyre.properties.float("theta_min", pi/2),
            pyre.properties.float("theta_max", pi*3/4),
            pyre.properties.float("fi_min", pi*3/4),
            pyre.properties.float("fi_max", pi*5/4),

            ]


# version
__id__ = "$Id: RegionalSphere.py,v 1.2 2003/07/23 22:00:57 tan2 Exp $"

# End of file
