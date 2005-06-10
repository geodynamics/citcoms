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

class Param(CitcomComponent):


    def __init__(self, name="param", facility="param"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.Param_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        file_vbcs = pyre.inventory.bool("file_vbcs", default=False)
        vel_bound_file = pyre.inventory.str("vel_bound_file", default="bvel.dat")

        mat_control = pyre.inventory.bool("mat_control", default=False)
        mat_file = pyre.inventory.str("mat_file", default="mat.dat")

        lith_age = pyre.inventory.bool("lith_age", default=False)
        lith_age_file = pyre.inventory.str("lith_age_file", default="age.dat")
        lith_age_time = pyre.inventory.bool("lith_age_time", default=False)
        lith_age_depth = pyre.inventory.float("lith_age_depth", default=0.0314)
        mantle_temp = pyre.inventory.float("mantle_temp", default=1.0)

        #DESCRIBE = pyre.inventory.bool("DESCRIBE", default=False)
        #BEGINNER = pyre.inventory.bool("BEGINNER", default=False)
        #VERBOSE = pyre.inventory.bool("VERBOSE", default=False)

        start_age = pyre.inventory.float("start_age", default=40.0)
        reset_startage = pyre.inventory.bool("reset_startage", default=False)



# version
__id__ = "$Id: Param.py,v 1.12 2005/06/10 02:23:21 leif Exp $"

# End of file
