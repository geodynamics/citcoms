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


    class Inventory(CitcomComponent.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.str("datafile", "test"),

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

            pyre.properties.bool("restart", False),
            pyre.properties.bool("post_p", False),
            pyre.properties.str("datafile_old", "test"),
            pyre.properties.int("solution_cycles_init", 100),
            pyre.properties.bool("zero_elapsed_time", True),

            pyre.properties.int("minstep", 1),
            pyre.properties.int("maxstep", 1),
            pyre.properties.int("maxtotstep", 1),
            pyre.properties.int("storage_spacing", 1),
            pyre.properties.int("cpu_limits_in_seconds", 360000000),

            pyre.properties.bool("stokes_flow_only", False),

            pyre.properties.float("inputdiffusivity", 1),

            pyre.properties.float("rayleigh", 1e+08),

            pyre.properties.float("Q0", 0.0),

            pyre.properties.bool("DESCRIBE", False),
            pyre.properties.bool("BEGINNER", False),
            pyre.properties.bool("VERBOSE", False),
            pyre.properties.bool("verbose", False),
            pyre.properties.bool("see_convergence", True),

            pyre.properties.float("start_age", 40.0),
            pyre.properties.bool("reset_startage", False)

            ]


# version
__id__ = "$Id: Param.py,v 1.6 2003/07/25 20:43:29 tan2 Exp $"

# End of file
