#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.facilities.Facility import Facility


class TSolver(Facility):


    def bind(self, configuration):
        componentName = configuration.get(self.name)

        # if the user didn't provide a setting, use the default
        if not componentName:
            component = self.component

        # these cases we want to treat specially
        elif componentName == "temperature_diffadv":
            import CitcomS.Advection_diffusion
            component = CitcomS.Advection_diffusion.temperature_diffadv()

        ## let the Facility handle unknown component names
        else:
            component = Facility.bind(self, configuration)

        return component




# version
__id__ = "$Id: TSolver.py,v 1.1 2003/07/03 23:40:21 ces74 Exp $"

# End of file
