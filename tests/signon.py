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

# version
__id__ = "$Id: signon.py,v 1.1 2003/03/24 01:46:37 tan2 Exp $"

#  End of file 
