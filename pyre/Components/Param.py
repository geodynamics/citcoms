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


        import pyre.properties


        inventory = [

            pyre.properties.bool("file_vbcs", False),
            pyre.properties.str("vel_bound_file", "bvel.dat"),

            pyre.properties.bool("mat_control", False),
            pyre.properties.str("mat_file", "mat.dat"),

            pyre.properties.bool("lith_age", False),
            pyre.properties.str("lith_age_file", "age.dat"),
            pyre.properties.bool("lith_age_time", False),
            pyre.properties.float("lith_age_depth", 0.0314),
            pyre.properties.float("mantle_temp", 1.0),

            pyre.properties.bool("tracer", False),
            pyre.properties.str("tracer_file", "tracer.dat"),

            #pyre.properties.bool("DESCRIBE", False),
            #pyre.properties.bool("BEGINNER", False),
            #pyre.properties.bool("VERBOSE", False),

            pyre.properties.float("start_age", 40.0),
            pyre.properties.bool("reset_startage", False)

            ]


# version
__id__ = "$Id: Param.py,v 1.9 2003/10/28 23:51:48 tan2 Exp $"

# End of file
