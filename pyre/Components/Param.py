#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Param(Component):


    def __init__(self):
        Component.__init__(self, "param", "param")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.Param_set_properties(self.inventory)
        return



    class Inventory(Component.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.str("datafile","test"),

            pyre.properties.bool("file_vbcs",False),
            pyre.properties.str("vel_bound_file","bvel.dat"),

            pyre.properties.bool("mat_control",False),
            pyre.properties.str("mat_file","mat.dat"),

            pyre.properties.bool("lith_age",False),
            pyre.properties.str("lith_age_file","age.dat"),
            pyre.properties.bool("lith_age_time",False),
            pyre.properties.float("lith_age_depth",0.314),
            pyre.properties.float("mantle_temp",1.0),

            pyre.properties.bool("tracer",False),
            pyre.properties.str("tracer_file",""),

            pyre.properties.bool("restart",False),
            pyre.properties.bool("post_p",False),
            pyre.properties.str("datafile_old","test"),
            pyre.properties.int("solution_cycles_init",100),
            pyre.properties.bool("zero_elapsed_time",True),

            pyre.properties.int("minstep",1),
            pyre.properties.int("maxstep",8001),
            pyre.properties.int("maxtotstep",8001),
            pyre.properties.int("storage_spacing",50),
            pyre.properties.int("cpu_limits_in_seconds",360000000),

            pyre.properties.bool("stokes_flow_only",False),

            pyre.properties.float("inputdiffusivity",0.001),

            pyre.properties.float("rayleigh",1.357e+08),

            pyre.properties.float("Q0",0.0),

            pyre.properties.bool("DESCRIBE",False),
            pyre.properties.bool("BEGINNER",False),
            pyre.properties.bool("VERBOSE",False),
            pyre.properties.bool("verbose",False),
            pyre.properties.bool("see_convergence",True),

            pyre.properties.float("start_age",40.0),
            pyre.properties.bool("reset_startage",False)

            ]


# version
__id__ = "$Id: Param.py,v 1.4 2003/07/23 22:00:57 tan2 Exp $"

# End of file
