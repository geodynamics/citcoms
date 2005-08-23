#!/usr/bin/env mpipython.exe
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

if __name__ == "__main__":

    import CitcomS 
#    import CitcomS.Full as Full
    import CitcomS.Regional as Regional
    from CitcomS.Components.Const import Const
    from CitcomS.Components.Visc import Visc

    import mpi
    Regional.citcom_init(mpi.world().handle())

    #const=Const()
    #print dir(const.inventory)
    #print const.inventory.radius.value
    #Regional.Const_set_properties(const.inventory)

    #visc=Visc()
    #print dir(visc.inventory) 
    #print visc.inventory.sdepv_expt
    #Regional.Visc_set_properties(visc.inventory)

    #Regional.set_signal()
    #Regional.set_convection_defaults()

    print "copyright information:"
    print "   ", CitcomS.copyright()

    print
    print "module information:"
#    print "  Full--"
#    print "    file:", Full.__file__
#    print "    doc:", Full.__doc__
#    print "    contents:", dir(Full)
    print
    print "  Regional--"
    print "    file:", Regional.__file__
    print "    doc:", Regional.__doc__
    print "    contents:", dir(Regional)


    print
#    print "Full.return1_test:    ", Full.return1_test()
    print "Regional.return1_test:", Regional.return1_test()

    print
    print "Time is %f" % Regional.CPU_time()

    import os, sys
    filename = sys.argv[1]
    print filename
    print "read_instructions:", Regional.read_instructions(filename)

    

# version
__id__ = "$Id: signon.py,v 1.12 2005/06/10 02:23:24 leif Exp $"

#  End of file 
