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

    import journal
    from journal import _journal as journalmodule

    print "copyright information:"
    print "   ", journal.copyright()
    print "   ", journalmodule.copyright()

    print
    print "module information:"
    print "    file:", journalmodule.__file__
    print "    doc:", journalmodule.__doc__
    print "    contents:", dir(journalmodule)

# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $"

#  End of file 
