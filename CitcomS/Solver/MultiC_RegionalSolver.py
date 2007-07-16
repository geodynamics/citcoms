#!/usr/bin/env python

#
#not sure whether this will work...
#

from CoupledRegionalSolver import CoupledRegionalSolver
from MultiC_Solver import MultiC_Solver
import journal

class MultiC_FullSolver(CoupledRegionalSolver):

    def __init__(self, name, facility="solver"):
        MultiC_Solver.__init__(self, name, facility)
        CoupledRegionalSolver.__init__(self, name, facility)
        return


    def initialize(self, application):
        MultiC_Solver.initialize(self, name, facility)
        CoupledRegionalSolver.initialize(self, name, facility)
        return
    

# version
__id__ = "Id"

# End of file
