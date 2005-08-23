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


class Advection_diffusion(CitcomComponent):


    def __init__(self, name, facility):
        CitcomComponent.__init__(self, name, facility)
        self.inventory.ADV = True

        self.inventory.adv_sub_iterations = 2
        self.inventory.maxadvtime = 10

        self.inventory.aug_lagr = True
        self.inventory.aug_number = 2.0e3
        return



    def setProperties(self):
        self.CitcomModule.Advection_diffusion_set_properties(self.all_variables, self.inventory)
        return



    def run(self,dt):
        self._solve(dt)
        return



    def setup(self):
        self.CitcomModule.set_convection_defaults(self.all_variables)
	self._been_here = False
	return


    def launch(self):
        self.CitcomModule.PG_timestep_init(self.all_variables)
        return

    #def fini(self):
	#return



    def _solve(self,dt):
        self.CitcomModule.PG_timestep_solve(self.all_variables, dt)
	return



    def stable_timestep(self):
        dt = self.CitcomModule.stable_timestep(self.all_variables)
        return dt



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory as prop


        inputdiffusivity = prop.float("inputdiffusivity", default=1)
        fixed_timestep = prop.float("fixed_timestep", default=0.0)
        finetunedt = prop.float("finetunedt", default=0.9)
        filter_temp = prop.bool("filter_temp", default=True)



# version
__id__ = "$Id: Advection_diffusion.py,v 1.23 2005/06/10 02:23:21 leif Exp $"

# End of file
