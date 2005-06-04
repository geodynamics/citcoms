#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.inventory.Facility import Facility
ScriptBinder = object


class VSolver(Facility):


    def __init__(self, name, default, binder=None):
        if not binder:
            binder = self.Binder()

        Facility.__init__(self, name, default=default) # , binder=binder)
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
__id__ = "$Id: VSolver.py,v 1.7 2005/06/03 21:51:45 leif Exp $"

# End of file
