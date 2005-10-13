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

    import elc
    from elc import elc as elcmodule

    print "copyright information:"
    print "   ", elc.copyright()
    print "   ", elcmodule.copyright()

    print
    print "module information:"
    print "    file:", elcmodule.__file__
    print "    doc:", elcmodule.__doc__
    print "    contents:", dir(elcmodule)


# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $"

#  End of file 
