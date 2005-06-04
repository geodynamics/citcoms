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


class Mesher(Facility):


    def __init__(self, name, default, binder=None):
        if not binder:
            binder = self.Binder()

        Facility.__init__(self, name, default=default) #, binder=binder)
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

	    import CitcomS.Components.Sphere as Sphere

            self._builtins = {
                "full-sphere": Sphere.fullSphere,
                "regional-sphere": Sphere.regionalSphere,
                }

            return




# version
__id__ = "$Id: Mesher.py,v 1.5 2005/06/03 21:51:45 leif Exp $"

# End of file
