#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomS.Components.CitcomComponent import CitcomComponent

class Sphere(CitcomComponent):



    def setup(self):
        return



    def run(self):
        start_time = self.CitcomModule.CPU_time()
        self.launch()

        import mpi
        if not mpi.world().rank:
            print "initialization time = %f" % \
                  (self.CitcomModule.CPU_time() - start_time)

	return



    def launch(self):
	raise NotImplementedError, "not implemented"
        return



    def setProperties(self):
        self.CitcomModule.Sphere_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties

        inventory = [

            pyre.properties.int("nprocx", 1),
            pyre.properties.int("nprocy", 1),
            pyre.properties.int("nprocz", 1),

            pyre.properties.bool("coor", False),
            pyre.properties.str("coor_file", "coor.dat"),

            pyre.properties.int("nodex", 9),
            pyre.properties.int("nodey", 9),
            pyre.properties.int("nodez", 9),
            pyre.properties.int("levels", 1),

            pyre.properties.float("radius_outer", 1.0),
            pyre.properties.float("radius_inner", 0.55),

	    # these parameters are for spherical harmonics output
	    # put them here temporalily
            pyre.properties.int("ll_max", 20),
            pyre.properties.int("nlong", 361),
            pyre.properties.int("nlati", 181),
            pyre.properties.int("output_ll_max", 20),

            ]



# version
__id__ = "$Id: Sphere.py,v 1.4 2003/10/28 23:51:48 tan2 Exp $"

# End of file
