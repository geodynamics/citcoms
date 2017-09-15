#!/usr/bin/env python


from CoupledRegionalSolver import CoupledRegionalSolver


class MultiC_RegionalSolver(CoupledRegionalSolver):


    def initialize(self, application):
        self.myPlus2 = application.myPlus2
        self.remotePlus2 = application.remotePlus2

        CoupledRegionalSolver.initialize(self, application)
        return


# version
__id__ = "$Id$"

# End of file
