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


class TSolver(Facility):


    def __init__(self, name, component):
        Facility.__init__(self, name,
                          default=component, binder=self.Binder())
        return



    class Binder(ScriptBinder):

        def bind(self,facility,value):
            try:
                return self._builtins[value]()
            except KeyError:
                pass

            return ScriptBinder.bind(self, facility, value)


        def __init__(self):
            ScriptBinder.__init__(self)

	    import CitcomS.Advection_diffusion as Advection_diffusion

            self._builtins = {
                "temp": Advection_diffusion.temperature_diffadv,
                }

            return            

# version
__id__ = "$Id: TSolver.py,v 1.2 2003/07/15 18:04:36 ces74 Exp $"

# End of file
