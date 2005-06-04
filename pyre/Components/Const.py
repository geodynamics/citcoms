#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent


class Const(CitcomComponent):


    def __init__(self, name="const", facility="const"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.Const_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory
        from pyre.units.length import m
        from pyre.units.length import km
        from pyre.units.mass import kg
        from pyre.units.time import s
        from pyre.units.temperature import K
        from pyre.units.pressure import Pa
        from pyre.units.energy import J

        #radius = pyre.inventory.float("radius", default=6371.0*km)
        #ref_density = pyre.inventory.float("ref_density", default=3500.0*kg/m**3)
        #thermdiff = pyre.inventory.float("thermdiff", default=1.0e-6*m**2/s)
        #gravacc = pyre.inventory.float("gravacc", default=10.0*m/s**2)
        #thermexp = pyre.inventory.float("thermexp", default=3.0e-5/K)
        #ref_visc = pyre.inventory.float("ref_visc", default=1.0e21*Pa*s)
        #heatcapacity = pyre.inventory.float("heatcapacity", default=1250.0*J/kg/K)
        #water_density = pyre.inventory.float("water_density", default=0.0*kg/m**3)

        #depth_lith = pyre.inventory.float("depth_lith", default=89e3*m)
        #depth_410 = pyre.inventory.float("depth_410", default=410e3*m)
        #depth_660 = pyre.inventory.float("depth_660", default=660e3*m)
        #depth_d_double_prime = pyre.inventory.float("depth_d_double_prime", default=2691e3*m)
        #depth_cmb = pyre.inventory.float("depth_cmb", default=2891e3*m)

	    # everything in SI units
        radius = pyre.inventory.float("radius", default=6371.0e3)
        layerd = pyre.inventory.float("layerd", default=6371.0e3)
        density = pyre.inventory.float("density", default=3500.0)
        thermdiff = pyre.inventory.float("thermdiff", default=1.0e-6)
        gravacc = pyre.inventory.float("gravacc", default=10.0)
        thermexp = pyre.inventory.float("thermexp", default=3.0e-5)
        refvisc = pyre.inventory.float("refvisc", default=1.0e21)
        cp = pyre.inventory.float("cp", default=1250.0)
        wdensity = pyre.inventory.float("wdensity", default=0.0)
        surftemp = pyre.inventory.float("surftemp", default=273.0)

        z_lith = pyre.inventory.float("z_lith", default=0.014)
        z_410 = pyre.inventory.float("z_410", default=0.06435)
        z_lmantle = pyre.inventory.float("z_lmantle", default=0.105)
        z_cmb = pyre.inventory.float("z_cmb", default=0.439)



# version
__id__ = "$Id: Const.py,v 1.11 2005/06/03 21:51:43 leif Exp $"

# End of file
