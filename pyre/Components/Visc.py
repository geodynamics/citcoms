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

class Visc(CitcomComponent):


    def __init__(self, name="visc", facility="visc"):
        CitcomComponent.__init__(self, name, facility)

        self.inventory.Viscosity = "system"
        self.inventory.visc_smooth_method = 3
        return



    def setProperties(self):
        inv = self.inventory
        inv.visc0 = map(float, inv.visc0)
        inv.viscE = map(float, inv.viscE)
        inv.viscT = map(float, inv.viscT)
        inv.viscZ = map(float, inv.viscZ)
        inv.sdepv_expt = map(float, inv.sdepv_expt)

        self.CitcomModule.Visc_set_properties(self.all_variables, inv)
        return



    def updateMaterial(self):
        self.CitcomModule.Visc_update_material(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        VISC_UPDATE = pyre.inventory.bool("VISC_UPDATE", default=True)

        num_mat = pyre.inventory.int("num_mat", default=4)
        visc0 = pyre.inventory.list("visc0", default=[1, 1, 1, 1])

        TDEPV = pyre.inventory.bool("TDEPV", default=False)
        rheol = pyre.inventory.int("rheol", default=3)
        viscE = pyre.inventory.list("viscE", default=[1, 1, 1, 1])
        viscT = pyre.inventory.list("viscT", default=[1, 1, 1, 1])
        viscZ = pyre.inventory.list("viscZ", default=[1, 1, 1, 1])

        SDEPV = pyre.inventory.bool("SDEPV", default=False)
        sdepv_expt = pyre.inventory.list("sdepv_expt", default=[1, 1, 1, 1])
        sdepv_misfit = pyre.inventory.float("sdepv_misfit", default=0.02)

        VMIN = pyre.inventory.bool("VMIN", default=False)
        visc_min = pyre.inventory.float("visc_min", default=1.0e-3)

        VMAX = pyre.inventory.bool("VMAX", default=False)
        visc_max = pyre.inventory.float("visc_max", default=1.0e3)


# version
__id__ = "$Id: Visc.py,v 1.14 2005/06/10 02:23:21 leif Exp $"

# End of file
