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

        import pyre.inventory


        nprocx = pyre.inventory.int("nprocx", default=1)
        nprocy = pyre.inventory.int("nprocy", default=1)
        nprocz = pyre.inventory.int("nprocz", default=1)

        coor = pyre.inventory.bool("coor", default=False)
        coor_file = pyre.inventory.str("coor_file", default="coor.dat")

        nodex = pyre.inventory.int("nodex", default=9)
        nodey = pyre.inventory.int("nodey", default=9)
        nodez = pyre.inventory.int("nodez", default=9)
        levels = pyre.inventory.int("levels", default=1)

        radius_outer = pyre.inventory.float("radius_outer", default=1.0)
        radius_inner = pyre.inventory.float("radius_inner", default=0.55)

	    # these parameters are for spherical harmonics output
	    # put them here temporalily
        ll_max = pyre.inventory.int("ll_max", default=20)
        nlong = pyre.inventory.int("nlong", default=361)
        nlati = pyre.inventory.int("nlati", default=181)
        output_ll_max = pyre.inventory.int("output_ll_max", default=20)




# version
__id__ = "$Id: Sphere.py,v 1.5 2005/06/03 21:51:45 leif Exp $"

# End of file
