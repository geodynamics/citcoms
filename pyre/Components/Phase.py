#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Phase(CitcomComponent):


    class Inventory(CitcomComponent.Inventory):


        import pyre.properties

        inventory = [

            pyre.properties.float("Ra_410", 0.0),
            pyre.properties.float("clapeyron410", 0.0235),
            pyre.properties.float("transT410", 0.78),
            pyre.properties.float("width410", 0.0058),

            pyre.properties.float("Ra_670", 0.0),
            pyre.properties.float("clapeyron670", -0.0235),
            pyre.properties.float("transT670", 0.78),
            pyre.properties.float("width670", 0.0058),

            pyre.properties.float("Ra_cmb", 0.0),
            pyre.properties.float("clapeyroncmb", -0.0235),
            pyre.properties.float("transTcmb", 0.875),
            pyre.properties.float("widthcmb", 0.0058)

            ]

# version
__id__ = "$Id: Phase.py,v 1.5 2003/07/25 20:43:29 tan2 Exp $"

# End of file
