#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Visc(CitcomComponent):


    class Inventory(CitcomComponent.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.str("Viscosity", "system"),
            pyre.properties.int("rheol", 3),
            pyre.properties.int("visc_smooth_method", 3),
            pyre.properties.bool("VISC_UPDATE", True),

            pyre.properties.int("num_mat", 4),
            pyre.properties.sequence("visc0", [ 1, 1, 1, 1]),

            pyre.properties.bool("TDEPV", False),
            pyre.properties.sequence("viscE", [1, 1, 1, 1]),
            pyre.properties.sequence("viscT", [1, 1, 1, 1]),

            pyre.properties.bool("SDEPV",False),
            pyre.properties.sequence("sdepv_expt", [1,1,1,1]),
            pyre.properties.float("sdepv_misfit", 0.02),

            pyre.properties.bool("VMIN", False),
            pyre.properties.float("visc_min", 1.0e-3),

            pyre.properties.bool("VMAX", False),
            pyre.properties.float("visc_max", 1.0e3)

            ]

# version
__id__ = "$Id: Visc.py,v 1.6 2003/07/25 20:43:29 tan2 Exp $"

# End of file
