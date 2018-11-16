#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

from Load import Load


class Bath(Load):


    class Inventory(Load.Inventory):

        import pyre.inventory
        
        from pyre.units.mass import kg
        from pyre.units.length import meter
        from pyre.units.volume import liter
        from pyre.units.pressure import atm

        ambient = pyre.inventory.dimensional("ambient", default=1.0*atm)
        surface = pyre.inventory.dimensional("surface", default=0.0*meter)
        density = pyre.inventory.dimensional("density", default=1.0*kg/liter)


    def updatePressure(self, boundary):

        ambient = self.ambient.value
        surface = self.surface.value
        density = self.density.value

        import pulse
        pulse.bath(boundary.mesh.handle(), boundary.pressure, ambient, surface, density)

        return


    def advance(self, dt):
        return
    

    def __init__(self):
        Load.__init__(self, "bath")
        self.ambient = None
        self.surface = None
        self.density = None
        return


    def _configure(self):
        self.ambient = self.inventory.ambient
        self.surface = self.inventory.surface
        self.density = self.inventory.density
        return


# version
__id__ = "$Id: Bath.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

#  End of file 
