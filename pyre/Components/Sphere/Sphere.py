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


    def __init__(self, name, facility, CitcomModule):
        # bind component method to facility method
        CitcomModule.mesher_set_properties = CitcomModule.Sphere_set_properties

        CitcomComponent.__init__(self, name, facility, CitcomModule)
	
	# will be overridden by child class
	self.inventory.nproc_surf = None
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
    
    

    class Inventory(CitcomComponent.Inventory):

        import pyre.properties
        from math import pi

        inventory = [

            pyre.properties.int("nprocx", 1),
            pyre.properties.int("nprocy", 1),
            pyre.properties.int("nprocz", 1),

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

            # used only in Regional version, not in Full version
            pyre.properties.float("theta_min", 1.5),
            pyre.properties.float("theta_max", 1.8),
            pyre.properties.float("fi_min", 0.0),
            pyre.properties.float("fi_max", 0.4),

            # these three parameters are not used anymore, will be removed in the future
            pyre.properties.float("dimenx", 1.0),
            pyre.properties.float("dimeny", 1.0),
            pyre.properties.float("dimenz", 1.0),


	    # these parameters are for spherical harmonics output
	    # put them here temporalily
            pyre.properties.int("ll_max", 20),
            pyre.properties.int("nlong", 361),
            pyre.properties.int("nlati", 181),
            pyre.properties.int("output_ll_max", 20),

            ]



# version
__id__ = "$Id: Sphere.py,v 1.1 2003/08/01 19:05:36 tan2 Exp $"

# End of file
