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

class Sphere(CitcomComponent):



    def setup(self):
        return



    def run(self):
        start_time = self.CitcomModule.CPU_time()
        self.launch()

        import mpi
        if not mpi.world().rank:
            import sys
            print >> sys.stderr, "initialization time = %f" % \
                  (self.CitcomModule.CPU_time() - start_time)

	return



    def launch(self):
	raise NotImplementedError, "not implemented"
        return



    def setProperties(self):
        self.CitcomModule.Sphere_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory


        nprocx = pyre.inventory.int("nprocx", default=1)
        nprocy = pyre.inventory.int("nprocy", default=1)
        nprocz = pyre.inventory.int("nprocz", default=1)

        coor = pyre.inventory.bool("coor", default=False)
        coor_file = pyre.inventory.str("coor_file", default="coor.dat")

        nodex = pyre.inventory.int("nodex", default=9)
        nodey = pyre.inventory.int("nodey", default=9)
        nodez = pyre.inventory.int("nodez", default=9)
        levels = pyre.inventory.int("levels", default=1)

        radius_outer = pyre.inventory.float("radius_outer", default=1.0)
        radius_inner = pyre.inventory.float("radius_inner", default=0.55)

	    # these parameters are for spherical harmonics output
	    # put them here temporalily
        ll_max = pyre.inventory.int("ll_max", default=20)
        nlong = pyre.inventory.int("nlong", default=361)
        nlati = pyre.inventory.int("nlati", default=181)
        output_ll_max = pyre.inventory.int("output_ll_max", default=20)




# version
__id__ = "$Id: Sphere.py,v 1.7 2005/06/15 18:27:42 tan2 Exp $"

# End of file
