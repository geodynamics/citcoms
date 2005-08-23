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

class Phase(CitcomComponent):


    def __init__(self, name="phase", facility="phase"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.Phase_set_properties(self.all_variables, self.inventory)
        return


    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory


        Ra_410 = pyre.inventory.float("Ra_410", default=0.0)
        clapeyron410 = pyre.inventory.float("clapeyron410", default=0.0235)
        transT410 = pyre.inventory.float("transT410", default=0.78)
        width410 = pyre.inventory.float("width410", default=0.0058)

        Ra_670 = pyre.inventory.float("Ra_670", default=0.0)
        clapeyron670 = pyre.inventory.float("clapeyron670", default=-0.0235)
        transT670 = pyre.inventory.float("transT670", default=0.78)
        width670 = pyre.inventory.float("width670", default=0.0058)

        Ra_cmb = pyre.inventory.float("Ra_cmb", default=0.0)
        clapeyroncmb = pyre.inventory.float("clapeyroncmb", default=-0.0235)
        transTcmb = pyre.inventory.float("transTcmb", default=0.875)
        widthcmb = pyre.inventory.float("widthcmb", default=0.0058)


# version
__id__ = "$Id: Phase.py,v 1.9 2005/06/10 02:23:21 leif Exp $"

# End of file
