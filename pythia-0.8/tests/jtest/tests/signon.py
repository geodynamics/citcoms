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

def test():
    import jtest
    from jtest import jtest as jtestmodule

    print "copyright information:"
    print "   ", jtest.copyright()
    print "   ", jtestmodule.copyright()

    print
    print "module information:"
    print "    file:", jtestmodule.__file__
    print "    doc:", jtestmodule.__doc__
    print "    contents:", dir(jtestmodule)

    return

# main

if __name__ == "__main__":
    test()


# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/18 17:01:42 aivazis Exp $"

#  End of file 
