#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class BC(Component):


    def __init__(self):
        Component.__init__(self, "bc", "bc")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.BC_set_prop(self.properties)
        return



    class Properties(Component.Properties):

        import pyre.properties


        __properties__ = Component.Properties.__properties__ + (

            pyre.properties.bool("topvbc",False),
            pyre.properties.float("topvbxval",0.0),
            pyre.properties.float("topvbyval",0.0),

            pyre.properties.bool("botvbc",False),
            pyre.properties.float("botvbxval",0.0),
            pyre.properties.float("botvbyval",0.0),

            pyre.properties.bool("toptbc",False),
            pyre.properties.float("toptbcval",0.0),

            pyre.properties.bool("bottbc",False),
            pyre.properties.float("bottbcval",0.0),


	    # these parameters are for 'lith_age',
	    # put them here temporalily
            pyre.properties.bool("temperature_bound_adj",False),
            pyre.properties.float("depth_bound_adj",0.157),
            pyre.properties.float("width_bound_adj",0.08727),

            )

# version
__id__ = "$Id: BC.py,v 1.2 2003/06/27 00:15:02 tan2 Exp $"

# End of file
