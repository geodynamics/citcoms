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


    def __init__(self, name, default, binder=None):
        if not binder:
            binder = self.Binder()

        Facility.__init__(self, name,
                          default=default, binder=binder)
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
__id__ = "$Id: VSolver.py,v 1.6 2003/08/29 20:40:22 tan2 Exp $"

# End of file
