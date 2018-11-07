#!/usr/bin/env mpipython.exe
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
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
__id__ = "$Id$"

#  End of file 
