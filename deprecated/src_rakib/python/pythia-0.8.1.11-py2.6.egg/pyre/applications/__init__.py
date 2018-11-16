#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Application import Application
from AppRunner import AppRunner
from ComponentHarness import ComponentHarness
from Script import Script
from Shell import Shell
from SimpleComponentHarness import SimpleComponentHarness
from SuperScript import SuperScript


def commandlineParser():
    from CommandlineParser import CommandlineParser
    return CommandlineParser()
    

def superCommandlineParser():
    from SuperCommandlineParser import SuperCommandlineParser
    return SuperCommandlineParser()
    

def start(argv=None, **kwds):
    """general-purpose entry point for applications"""
    cls = kwds.get('applicationClass')
    kwds = dict(**kwds)
    kwds['argv'] = argv
    app = cls()
    shell = Shell(app)
    shell.run(**kwds)
    return 0


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
