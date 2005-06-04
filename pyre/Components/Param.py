#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Param(CitcomComponent):


    def __init__(self, name="param", facility="param"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.Param_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        file_vbcs = pyre.inventory.bool("file_vbcs", default=False)
        vel_bound_file = pyre.inventory.str("vel_bound_file", default="bvel.dat")

        mat_control = pyre.inventory.bool("mat_control", default=False)
        mat_file = pyre.inventory.str("mat_file", default="mat.dat")

        lith_age = pyre.inventory.bool("lith_age", default=False)
        lith_age_file = pyre.inventory.str("lith_age_file", default="age.dat")
        lith_age_time = pyre.inventory.bool("lith_age_time", default=False)
        lith_age_depth = pyre.inventory.float("lith_age_depth", default=0.0314)
        mantle_temp = pyre.inventory.float("mantle_temp", default=1.0)

        #DESCRIBE = pyre.inventory.bool("DESCRIBE", default=False)
        #BEGINNER = pyre.inventory.bool("BEGINNER", default=False)
        #VERBOSE = pyre.inventory.bool("VERBOSE", default=False)

        start_age = pyre.inventory.float("start_age", default=40.0)
        reset_startage = pyre.inventory.bool("reset_startage", default=False)



# version
__id__ = "$Id: Param.py,v 1.11 2005/06/03 21:51:44 leif Exp $"

# End of file
