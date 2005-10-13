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

    import tabulator
    from tabulator import _tabulator as tabulatormodule

    print "copyright information:"
    print "   ", tabulator.copyright()
    print "   ", tabulatormodule.copyright()

    print
    print "module information:"
    print "    file:", tabulatormodule.__file__
    print "    doc:", tabulatormodule.__doc__
    print "    contents:", dir(tabulatormodule)


# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

#  End of file 
