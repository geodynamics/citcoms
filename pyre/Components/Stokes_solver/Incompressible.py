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

from CitcomS.Components.CitcomComponent import CitcomComponent


class Incompressible(CitcomComponent):


    def __init__(self, name, facility):
        CitcomComponent.__init__(self, name, facility)

        return



    def run(self):
        self.CitcomModule.general_stokes_solver(self.all_variables)
	return



    def setup(self):
        if self.inventory.Solver == "cgrad":
            self.CitcomModule.set_cg_defaults(self.all_variables)
        elif self.inventory.Solver == "multigrid":
            self.CitcomModule.set_mg_defaults(self.all_variables)
        elif self.inventory.Solver == "multigrid-el":
            self.CitcomModule.set_mg_el_defaults(self.all_variables)
        return



    def launch(self):
        self.CitcomModule.general_stokes_solver_setup(self.all_variables)
        return



    #def fini(self):
	#return



    def setProperties(self):
        self.CitcomModule.Incompressible_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory as prop


        Solver = prop.str("Solver", default="cgrad",
                 validator=prop.choice(["cgrad",
                                        "multigrid",
                                        "multigrid-el"]))
        node_assemble = prop.bool("node_assemble", default=True)
        precond = prop.bool("precond", default=True)

        accuracy = prop.float("accuracy", default=1.0e-6)
        tole_compressibility = prop.float("tole_compressibility", default=1.0e-7)
        mg_cycle = prop.int("mg_cycle", default=1)
        down_heavy = prop.int("down_heavy", default=3)
        up_heavy = prop.int("up_heavy", default=3)

        vlowstep = prop.int("vlowstep", default=1000)
        vhighstep = prop.int("vhighstep", default=3)
        piterations = prop.int("piterations", default=1000)


# version
__id__ = "$Id: Incompressible.py,v 1.17 2005/06/10 02:23:22 leif Exp $"

# End of file
