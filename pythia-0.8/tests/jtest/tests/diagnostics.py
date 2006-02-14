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
    # force the initialization
    journal.journal()

    from jtest import jtest

    print " ** testing informationals"
    info = journal.info("jtest")
    info.activate()
    info.log("this is an info from python")
    jtest.info("jtest")

    print " ** testing warnings"
    warning = journal.warning("jtest")
    warning.log("this a warning from python")
    #jtest.warning("jtest")

    print " ** testing errors"
    error = journal.error("jtest")
    error.log("this an error from python")
    #jtest.error("jtest")

    return

# main

if __name__ == "__main__":
    test()


# version
__id__ = "$Id: diagnostics.py,v 1.1.1.1 2005/03/18 17:01:42 aivazis Exp $"

#  End of file 
