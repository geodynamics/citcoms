#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent


class Tracer(CitcomComponent):


    def __init__(self, name="tracer", facility="tracer"):
        CitcomComponent.__init__(self, name, facility)
        return



    def run(self):
        self.CitcomModule.Tracer_tracer_advection(self.all_variables)
        return



    def setProperties(self):
        self.CitcomModule.Tracer_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.properties

        inventory = [

            pyre.properties.bool("tracer", False),
            pyre.properties.str("tracer_file", "tracer.dat"),

            ]


# version
__id__ = "$Id: Tracer.py,v 1.2 2005/01/19 02:02:27 tan2 Exp $"

# End of file
