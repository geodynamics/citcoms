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

_NAME = 'hello'

def test():
    import pyre.inventory

    # deal with the commandline
    help, extras, registry = processCommandline()

    # create the curator
    curator = pyre.inventory.curator(_NAME)

    # configure the curator with the commandline arguments
    curator.config(registry)

    # add some depositories
    curator.depositories += curator.createPrivateDepositories(_NAME)
    
    # get some traits
    # registry = curator.getTraits(_NAME, encoding='pml')

    print " ++ curator:", curator
    print " ++ codecs:", curator.codecs.keys()
    # print " ++ depositories:", [depository.name for depository in curator.depositories]

    registry = None
    pml = curator.codecs['pml']
    print " ++ looking for 'inventory'"
    for new, locator in curator.loadSymbol(
        tag=_NAME, codec=pml, address=[_NAME], symbol='inventory', errorHandler=handler): 
        
        registry = new.update(registry)

    print " ++ registry:", registry.render()


    registry = curator.getTraits(_NAME)
    print " ++ registry:", registry.render()
    print " ++ trait requests:", curator._traitRequests

    return


def handler(symbol, filename, message):
    print " ##  %s: %s: %s" % (symbol, filename, message)
    return


# helpers
def processCommandline():
    import pyre.inventory
    registry = pyre.inventory.registry(_NAME)

    import pyre.applications
    parser = pyre.applications.commandlineParser()

    help, args = parser.parse(registry)
    return help, args, registry

# main
if __name__ == "__main__":
    test()


# version
__id__ = "$Id: curator.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
