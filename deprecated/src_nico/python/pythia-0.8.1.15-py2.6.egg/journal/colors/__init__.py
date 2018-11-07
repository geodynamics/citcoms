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


from ColorScheme import ColorScheme


# facilities and components

def colorScheme(name, **kwds):
    from pyre.inventory.Facility import Facility
    kwds['factory'] = kwds.get('factory', ColorScheme)
    kwds['vault'] = kwds.get('vault', ['colors'])
    kwds['family'] = kwds.get('family', 'colorScheme')
    return Facility(name, **kwds)


# end of file 
