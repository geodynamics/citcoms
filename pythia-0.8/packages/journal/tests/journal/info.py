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
    print journal.copyright()

    info = journal.info("info")
    print "state of %s(%s): %s" % (info.facility, info.severity, info.state)
    info.state = True

    info = journal.info("info")
    print "state of %s(%s): %s" % (info.facility, info.severity, info.state)
    info.log("hello")

    print "info facilities:", journal.infoIndex().facilities()


# version
__id__ = "$Id: info.py,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $"

#  End of file 
