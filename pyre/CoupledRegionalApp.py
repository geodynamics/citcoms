#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

import journal
from CitcomSRegional import CitcomSRegional


class CoupledRegional(CitcomSRegional):


    def run_simulation(self):
        mesher = self.inventory.mesher
        mesher.setup()

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

        carrier = self.inventory.carrier
        carrier.setup()

        mesher.run()

	vsolver.run()

        self._output(self._cycles)

	while self._cycles < self.inventory.param.inventory.maxstep:
	    self._cycles += 1

	    tsolver.run()
	    vsolver.run()

            if not (self._cycles %
                    self.inventory.param.inventory.storage_spacing):
                self._output(self._cycles)

        return



    class Inventory(CitcomSRegional.Inventory):

        import pyre.facilities
        from Components.Carrier import Carrier

        inventory = [

            pyre.facilities.facility("carrier",
                                     default=Carrier("carrier", "carrier", CitcomModule)),

            ]


# version
__id__ = "$Id: CoupledRegionalApp.py,v 1.2 2003/08/27 20:52:46 tan2 Exp $"

# End of file
