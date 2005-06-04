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


        import pyre.inventory


        tracer = pyre.inventory.bool("tracer", default=False)
        tracer_file = pyre.inventory.str("tracer_file", default="tracer.dat")



# version
__id__ = "$Id: Tracer.py,v 1.3 2005/06/03 21:51:44 leif Exp $"

# End of file
