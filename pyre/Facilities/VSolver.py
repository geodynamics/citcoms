#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.facilities.Facility import Facility
from pyre.facilities.ScriptBinder import ScriptBinder


class VSolver(Facility):


    def __init__(self, name, component):
        Facility.__init__(self, name,
                          default=component, binder=self.Binder())
        return



    class Binder(ScriptBinder):


        def bind(self, facility, value):
            try:
                return self._builtins[value]()
            except KeyError:
                pass

            return ScriptBinder.bind(self, facility, value)


        def __init__(self):
            ScriptBinder.__init__(self)

	    import CitcomS.Components.Stokes_solver as Stokes

            self._builtins = {
                "incomp-newtonian": Stokes.incompressibleNewtonian,
                "incomp-non-newtonian": Stokes.incompressibleNonNewtonian,
                }

            return




# version
__id__ = "$Id: VSolver.py,v 1.5 2003/08/22 22:16:05 tan2 Exp $"

# End of file
