#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.facilities.Facility import Facility


class VSolver(Facility):


    def bind(self, configuration):
        componentName = configuration.get(self.name)

        # if the user didn't provide a setting, use the default
        if not componentName:
            component = self.component

        # these cases we want to treat specially
        elif componentName == "imcomp-newtonian":
            import CitcomS.Stokes_solver
            component = CitcomS.Stokes_solver.imcompressibleNewtonian()

        elif componentName == "imcomp-non-newtonian":
	    import CitcomS.Stokes_solver
            component = CitcomS.Stokes_solver.imcompressibleNonNewtonian()

        ## let the Facility handle unknown component names
        else:
            component = Facility.bind(configuration)

        return component




# version
__id__ = "$Id: VSolver.py,v 1.1 2003/06/23 20:43:32 tan2 Exp $"

# End of file
