#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomApp import CitcomApp
import Full as CitcomModule
import journal


class FullApp(CitcomApp):


    def __init__(self, name):
	CitcomApp.__init__(self, name)
	self.CitcomModule = CitcomModule
        return



    class Inventory(CitcomApp.Inventory):

        import pyre.facilities

        # facilities
        from Facilities.Mesher import Mesher

        # component modules
        import Components.Sphere as Sphere

        inventory = [

            Mesher("mesher", Sphere.fullSphere(CitcomModule)),

            ]



# version
__id__ = "$Id: FullApp.py,v 1.2 2003/08/22 22:18:41 tan2 Exp $"

# End of file
