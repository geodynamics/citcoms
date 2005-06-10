#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def coupler(name="coupler", facility="coupler"):
    return Coupler(name, facility)


from pyre.components.Component import Component


class Coupler(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.exchanger = None
        return



    def initialize(self, solver):
        # exchanger could be either a FineGridExchanger (FGE)
        # or a CoarseGridExchanger (CGE)
        self.exchanger = solver.exchanger
        self.exchanger.initialize(solver)
        return



    def launch(self, solver):
        self.exchanger.launch(solver)
        return



    def initTemperature(self):
        # send initial temperature field from CGE to FGE
        self.exchanger.initTemperature()
        return



    def preVSolverRun(self):
        self.exchanger.preVSolverRun()
        return



    def postVSolverRun(self):
        self.exchanger.postVSolverRun()
        return



    def newStep(self):
        self.exchanger.NewStep()
        return



    def applyBoundaryConditions(self):
        self.exchanger.applyBoundaryConditions()
        return



    def stableTimestep(self, dt):
        dt = self.exchanger.stableTimestep(dt)
        return dt



    def endTimestep(self, steps, done):
        done = self.exchanger.endTimestep(steps, done)
        return done


# version
__id__ = "$Id: Coupler.py,v 1.12 2005/06/10 02:23:20 leif Exp $"

# End of file
