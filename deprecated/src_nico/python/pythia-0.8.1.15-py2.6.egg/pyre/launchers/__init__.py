#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Launcher import Launcher


# facilities and components

def facility(name, **kwds):
    from pyre.inventory.Facility import Facility
    kwds['factory'] = kwds.get('factory', Launcher)
    kwds['vault'] = kwds.get('vault', ['launchers'])
    kwds['family'] = kwds.get('family', 'launcher')
    return Facility(name, **kwds)


# end of file 
