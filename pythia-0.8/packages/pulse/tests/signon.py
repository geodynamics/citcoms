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

    import pulse
    from pulse import pulse as pulsemodule

    print "copyright information:"
    print "   ", pulse.copyright()
    print "   ", pulsemodule.copyright()

    print
    print "module information:"
    print "    file:", pulsemodule.__file__
    print "    doc:", pulsemodule.__doc__
    print "    contents:", dir(pulsemodule)

# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

#  End of file 
