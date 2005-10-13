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
    info.flip()
    info.log("hello")

    warning = journal.warning("warning")
    warning.log("hello")

    error = journal.error("error")
    error.log("hello")

    debug = journal.debug("debug")
    debug.flip()
    debug.log("hello")

    firewall = journal.firewall("firewall")
    firewall.log("hello")



# version
__id__ = "$Id: diagnostics.py,v 1.1.1.1 2005/03/08 16:13:53 aivazis Exp $"

#  End of file 
