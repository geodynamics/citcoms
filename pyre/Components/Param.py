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


    class Properties(Component.Properties):


        import pyre.properties
        import os

        __properties__ = Component.Properties.__properties__ + (
            pyre.properties.string("datafile","test"),

            pyre.properties.bool("file_vbcs",False),
            pyre.properties.string("vel_bound_file","bvel.dat"),

            pyre.properties.bool("mat_control",False),
            pyre.properties.string("mat_file","mat.dat"),
            
            pyre.properties.bool("lith_age",False),
            pyre.properties.string("lith_age_file","age.dat"),
            pyre.properties.bool("lith_age_time",False),
            pyre.properties.float("lith_age_depth",0.314),
            pyre.properties.float("mantle_temp",1.0),

            pyre.properties.bool("tracer",False),
            pyre.properties.string("tracer_file",""),
            
            pyre.properties.bool("restart",False),
            pyre.properties.bool("post_p",False),
            pyre.properties.string("datafile_old","test"),
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
            pyre.properties.bool("reset_startage",False),
            )


# version
__id__ = "$Id: Param.py,v 1.1 2003/06/11 23:02:09 tan2 Exp $"

# End of file 
