#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from pyre.facilities.Facility import Facility
from pyre.facilities.ScriptBinder import ScriptBinder


class Solver(Facility):


    def __init__(self, name, default, binder=self.Binder()):
        Facility.__init__(self, name,
                          default, binder)
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

            import CitcomS.Solver as SolverComponent

            self._builtins = {
                "full": SolverComponent.fullSolver,
                "regional": SolverComponent.regionalSolver,
                }

            return




# version
__id__ = "$Id: Solver.py,v 1.1 2003/08/29 19:46:38 tan2 Exp $"

# End of file
