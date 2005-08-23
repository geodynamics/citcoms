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

class BC(CitcomComponent):


    def __init__(self, name="bc", facility="bc"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.BC_set_properties(self.all_variables, self.inventory)
        return



    def updatePlateVelocity(self):
        self.CitcomModule.BC_update_plate_velocity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory



        side_sbcs = pyre.inventory.bool("side_sbcs", default=False)
        pseudo_free_surf = pyre.inventory.bool("pseudo_free_surf", default=False)

        topvbc = pyre.inventory.int("topvbc", default=0)
        topvbxval = pyre.inventory.float("topvbxval", default=0.0)
        topvbyval = pyre.inventory.float("topvbyval", default=0.0)

        botvbc = pyre.inventory.int("botvbc", default=0)
        botvbxval = pyre.inventory.float("botvbxval", default=0.0)
        botvbyval = pyre.inventory.float("botvbyval", default=0.0)

        toptbc = pyre.inventory.int("toptbc", default=True)
        toptbcval = pyre.inventory.float("toptbcval", default=0.0)

        bottbc = pyre.inventory.int("bottbc", default=True)
        bottbcval = pyre.inventory.float("bottbcval", default=1.0)


	    # these parameters are for 'lith_age',
	    # put them here temporalily
        temperature_bound_adj = pyre.inventory.bool("temperature_bound_adj", default=False)
        depth_bound_adj = pyre.inventory.float("depth_bound_adj", default=0.157)
        width_bound_adj = pyre.inventory.float("width_bound_adj", default=0.08727)


# version
__id__ = "$Id: BC.py,v 1.15 2005/06/10 02:23:20 leif Exp $"

# End of file
