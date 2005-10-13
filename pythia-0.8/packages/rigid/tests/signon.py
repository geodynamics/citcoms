#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

if __name__ == "__main__":

    import rigid
    from rigid import rigid as rigidmodule

    print "copyright information:"
    print "   ", rigid.copyright()
    print "   ", rigidmodule.copyright()

    print
    print "module information:"
    print "    file:", rigidmodule.__file__
    print "    doc:", rigidmodule.__doc__
    print "    contents:", dir(rigidmodule)


# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/08 16:13:58 aivazis Exp $"

#  End of file 
