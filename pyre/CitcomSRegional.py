#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Citcom import Citcom
import Regional as CitcomModule
import journal


class CitcomSRegional(Citcom):


    def __init__(self, name="regional", facility="citcom"):
	Citcom.__init__(self, name, facility)
	self.CitcomModule = CitcomModule
        return



    class Inventory(Citcom.Inventory):

        # facilities
        from CitcomS.Facilities.Mesher import Mesher

        # component modules
        import CitcomS.Components.Sphere as Sphere


        inventory = [

            Mesher("mesher", Sphere.regionalSphere("regional-sphere")),

            ]



# version
__id__ = "$Id: CitcomSRegional.py,v 1.29 2003/08/27 20:52:46 tan2 Exp $"

# End of file
