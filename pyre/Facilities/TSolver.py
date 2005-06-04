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


class TSolver(Facility):


    def __init__(self, name, default, binder=None):
        if not binder:
            binder = self.Binder()

        Facility.__init__(self, name, default=default) #, binder=binder)
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

	    import CitcomS.Components.Advection_diffusion as Advection_diffusion

            self._builtins = {
                "temp": Advection_diffusion.temperature_diffadv,
                }

            return

# version
__id__ = "$Id: TSolver.py,v 1.5 2005/06/03 21:51:45 leif Exp $"

# End of file
