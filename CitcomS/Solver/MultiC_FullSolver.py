#!/usr/bin/env python

#
#not sure whether this will work...
#

from CoupledFullSolver import CoupledFullSolver
from MultiC_Solver import MultiC_Solver
import journal

class MultiC_FullSolver(CoupledFullSolver):

    def __init__(self, name, facility="solver"):
        MultiC_Solver.__init__(self, name, facility)
        CoupledFullSolver.__init__(self, name, facility)
        return


    def initialize(self, application):
        MultiC_Solver.initialize(self, name, facility)
        CoupledFullSolver.initialize(self, name, facility)
        return
    

# version
__id__ = "Id"

# End of file
