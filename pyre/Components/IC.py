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

from CitcomComponent import CitcomComponent

class IC(CitcomComponent):


    def __init__(self, name="ic", facility="ic"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        inv = self.inventory
        inv.perturbmag = map(float, inv.perturbmag)
        inv.perturbl = map(float, inv.perturbl)
        inv.perturbm = map(float, inv.perturbm)
        inv.blob_center = map(float, inv.blob_center)

        self.CitcomModule.IC_set_properties(self.all_variables, inv)
        return



    def launch(self):
        self.initTemperature()
        self.initPressure()
        self.initVelocity()
        self.initViscosity()
        return



    def initTemperature(self):
        if self.inventory.restart:
            self.CitcomModule.restartTemperature(self.all_variables)
        else:
            self.CitcomModule.constructTemperature(self.all_variables)
        return



    def initPressure(self):
        self.CitcomModule.initPressure(self.all_variables)
        return



    def initVelocity(self):
        self.CitcomModule.initVelocity(self.all_variables)
        return



    def initViscosity(self):
        self.CitcomModule.initViscosity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        restart = pyre.inventory.bool("restart", default=False)
        post_p = pyre.inventory.bool("post_p", default=False)
        solution_cycles_init = pyre.inventory.int("solution_cycles_init", default=0)
        zero_elapsed_time = pyre.inventory.bool("zero_elapsed_time", default=True)

        tic_method = pyre.inventory.int("tic_method", default=0,
                            validator=pyre.inventory.choice([0, 1, 2]))

        half_space_age = pyre.inventory.float("half_space_age", default=40,
                              validator=pyre.inventory.greater(1e-3))

        num_perturbations = pyre.inventory.int("num_perturbations", default=1,
                            validator=pyre.inventory.less(255))
        perturbmag = pyre.inventory.list("perturbmag", default=[0.05])
        perturbl = pyre.inventory.list("perturbl", default=[1])
        perturbm = pyre.inventory.list("perturbm", default=[1])
        perturblayer = pyre.inventory.slice("perturblayer", default=[5])


        blob_center = pyre.inventory.list("blob_center", default=[-999., -999., -999.])
        blob_radius = pyre.inventory.float("blob_radius", default=[0.063])
        blob_dT = pyre.inventory.float("blob_dT", default=[0.18])


# version
__id__ = "$Id: IC.py,v 1.17 2005/06/10 02:23:21 leif Exp $"

# End of file
