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


    def setProperties(self):
        self.CitcomModule.Const_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.properties
        from pyre.units.length import m
        from pyre.units.length import km
        from pyre.units.mass import kg
        from pyre.units.time import s
        from pyre.units.temperature import K
        from pyre.units.pressure import Pa
        from pyre.units.energy import J

        inventory = [
            #pyre.properties.float("radius", 6371.0*km),
            #pyre.properties.float("ref_density", 3500.0*kg/m**3),
            #pyre.properties.float("thermdiff", 1.0e-6*m**2/s),
            #pyre.properties.float("gravacc", 10.0*m/s**2),
            #pyre.properties.float("thermexp", 3.0e-5/K),
            #pyre.properties.float("ref_visc", 1.0e21*Pa*s),
            #pyre.properties.float("heatcapacity", 1250.0*J/kg/K),
            #pyre.properties.float("water_density", 0.0*kg/m**3),

            #pyre.properties.float("depth_lith", 89e3*m),
            #pyre.properties.float("depth_410", 410e3*m),
            #pyre.properties.float("depth_660", 660e3*m),
            #pyre.properties.float("depth_d_double_prime", 2691e3*m),
            #pyre.properties.float("depth_cmb", 2891e3*m)

	    # everything in SI units
            pyre.properties.float("radius", 6371.0e3),
            pyre.properties.float("layerd", 6371.0e3),
            pyre.properties.float("density", 3500.0),
            pyre.properties.float("thermdiff", 1.0e-6),
            pyre.properties.float("gravacc", 10.0),
            pyre.properties.float("thermexp", 3.0e-5),
            pyre.properties.float("refvisc", 1.0e21),
            pyre.properties.float("cp", 1250.0),
            pyre.properties.float("wdensity", 0.0),

            pyre.properties.float("depth_lith", 89e3),
            pyre.properties.float("depth_410", 410e3),
            pyre.properties.float("depth_660", 660e3),
            #pyre.properties.float("depth_d_double_prime", 2691e3),
            pyre.properties.float("depth_cmb", 2691e3) # this is used as the D" phase change depth

            ]


# version
__id__ = "$Id: Const.py,v 1.7 2003/08/27 20:52:47 tan2 Exp $"

# End of file
