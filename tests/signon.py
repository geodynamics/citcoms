#!/usr/bin/env mpipython.exe
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2002 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

if __name__ == "__main__":

    import CitcomS 
    import CitcomS.Full as Full
    import CitcomS.Regional as Regional

    print "copyright information:"
    print "   ", CitcomS.copyright()

    print
    print "module information:"
    print "  Full--"
    print "    file:", Full.__file__
    print "    doc:", Full.__doc__
    print "    contents:", dir(Full)
    print
    print "  Regional--"
    print "    file:", Regional.__file__
    print "    doc:", Regional.__doc__
    print "    contents:", dir(Regional)


    print
    print "Full.return1_test:    ", Full.return1_test()
    print "Regional.return1_test:", Regional.return1_test()

    import mpi
    world = mpi.world()
    print
    print "Citcom_Init: return", Regional.Citcom_Init(world.size,
                                                             world.rank)

    import os, sys
    filename = sys.argv[1]
    print filename
    print "read_instructions:", Regional.read_instructions(filename)

    

# version
__id__ = "$Id: signon.py,v 1.5 2003/04/10 23:40:55 tan2 Exp $"

#  End of file 
