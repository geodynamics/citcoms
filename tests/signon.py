#!/usr/bin/env python
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
    import CitcomS.CitcomSFull as CitcomSFull
    import CitcomS.CitcomSRegional as CitcomSRegional

    print "copyright information:"
    print "   ", CitcomS.copyright()

    print
    print "module information:"
    print "  CitcomSFull--"
    print "    file:", CitcomSFull.__file__
    print "    doc:", CitcomSFull.__doc__
    print "    contents:", dir(CitcomSFull)
    print
    print "  CitcomSRegional--"
    print "    file:", CitcomSRegional.__file__
    print "    doc:", CitcomSRegional.__doc__
    print "    contents:", dir(CitcomSRegional)


    print
    print "CitcomSFull.return1_test:    ", CitcomSFull.return1_test()
    print "CitcomSRegional.return1_test:", CitcomSRegional.return1_test()

    print
    print "Citcom_Init: return", CitcomSRegional.Citcom_Init()

    import os
    filename = '%s/../packages/CitcomS/examples/Regional/input.sample' % os.environ["BLD_ROOT"]
    print filename
    print "read_instructions:", CitcomSRegional.read_instructions(filename)

# version
__id__ = "$Id: signon.py,v 1.3 2003/04/05 23:51:35 tan2 Exp $"

#  End of file 
