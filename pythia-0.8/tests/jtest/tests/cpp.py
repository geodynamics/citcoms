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
    import journal
    from jtest import jtest

    print " ** testing C++ informationals"
    info = journal.info("jtest")
    info.activate()
    jtest.info("jtest")

    print " ** testing C++ warnings"
    warning = journal.warning("jtest")
    #warning.deactivate()
    warning = jtest.warning("jtest")

    print " ** testing C++ errors"
    error = journal.error("jtest")
    #error.deactivate()
    jtest.error("jtest")

    return

# main

if __name__ == "__main__":
    test()


# version
__id__ = "$Id: cpp.py,v 1.1.1.1 2005/03/18 17:01:42 aivazis Exp $"

#  End of file 
