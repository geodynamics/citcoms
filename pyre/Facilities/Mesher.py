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


class Mesher(Facility):


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

	    import CitcomS.Components.Sphere as Sphere

            self._builtins = {
                "full-sphere": Sphere.fullSphere,
                "regional-sphere": Sphere.regionalSphere,
                }

            return




# version
__id__ = "$Id: Mesher.py,v 1.3 2003/08/01 19:05:36 tan2 Exp $"

# End of file
