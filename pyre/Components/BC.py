#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class BC(CitcomComponent):


    def __init__(self, name="bc", facility="bc"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.BC_set_properties(self.all_variables, self.inventory)
        return



    def updatePlateVelocity(self):
        self.CitcomModule.BC_update_plate_velocity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties


        inventory = [

            pyre.properties.bool("side_sbcs", False),

            pyre.properties.int("topvbc", 0),
            pyre.properties.float("topvbxval", 0.0),
            pyre.properties.float("topvbyval", 0.0),

            pyre.properties.int("botvbc", 0),
            pyre.properties.float("botvbxval", 0.0),
            pyre.properties.float("botvbyval", 0.0),

            pyre.properties.int("toptbc", True),
            pyre.properties.float("toptbcval", 0.0),

            pyre.properties.int("bottbc", True),
            pyre.properties.float("bottbcval", 1.0),


	    # these parameters are for 'lith_age',
	    # put them here temporalily
            pyre.properties.bool("temperature_bound_adj", False),
            pyre.properties.float("depth_bound_adj", 0.157),
            pyre.properties.float("width_bound_adj", 0.08727)

            ]

# version
__id__ = "$Id: BC.py,v 1.12 2004/05/26 23:55:38 tan2 Exp $"

# End of file
