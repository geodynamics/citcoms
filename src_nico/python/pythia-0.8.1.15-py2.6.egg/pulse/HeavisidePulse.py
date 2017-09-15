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


class HeavisidePulse(Load):


    class Inventory(Load.Inventory):

        import pyre.inventory
        from pyre.units.SI import meter, second, pascal

        amplitude = pyre.inventory.dimensional("amplitude", default=0.0*pascal)
        position = pyre.inventory.dimensional(
            "position", default=(0.0*meter, 0.0*meter, 0.0*meter))
        velocity = pyre.inventory.dimensional(
            "velocity", default=(0.0*meter/second, 0.0*meter/second, 0.0*meter/second))


    def updatePressure(self, boundary):

        amplitude = self.amplitude.value
        position = [ r.value for r in self.position ]
        velocity = [ v.value for v in self.velocity ]

        import pulse
        pulse.heaviside(boundary.mesh.handle(), boundary.pressure, amplitude, position, velocity)

        return


    def advance(self, dt):
        x, y, z = self.position
        v_x, v_y, v_z = self.velocity

        x += dt * v_x
        y += dt * v_y
        z += dt * v_z

        self.position = (x, y, z)

        self._info.log("pulse front: (%s, %s, %s)" % (x, y, z))
        self._info.log("pulse velocity: (%s, %s, %s)" % (v_x, v_y, v_z))

        return
    

    def __init__(self):
        Load.__init__(self, "heaviside")
        self.position = None
        self.velocity = None
        self.amplitude = None
        return


    def _configure(self):
        self.position = self.inventory.position
        self.velocity = self.inventory.velocity
        self.amplitude = self.inventory.amplitude
        return


# version
__id__ = "$Id: HeavisidePulse.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

#  End of file 
