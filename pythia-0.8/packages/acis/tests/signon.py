#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

if __name__ == "__main__":

    import acis
    from acis import acis as acismodule

    print "copyright information:"
    print "   ", acis.copyright()
    print "   ", acismodule.copyright()

    print
    print "module information:"
    print "    file:", acismodule.__file__
    print "    doc:", acismodule.__doc__
    print "    contents:", dir(acismodule)

# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/08 16:13:38 aivazis Exp $"

#  End of file 
