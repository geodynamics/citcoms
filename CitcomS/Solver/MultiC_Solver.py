#!/usr/bin/env python

#
#some additional attributes used in MultiCoupled solvers
#
#can be think as a patch to CoupledSolver.py
#


import journal

class MultiC_Solver:

    def __init__(self, name, facility="solver"):
        self.myPlus2 = []
        self.remotePlus2 = []
        return


    def initialize(self, application):
        self.myPlus2 = application.myPlus2
        self.remotePlus2 = application.remotePlus2
        return
    

# version
__id__ = "Id"

# End of file
    
