#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def temperature_diffadv(CitcomModule):
    from Temperature_diffadv import Temperature_diffadv
    return Temperature_diffadv('temp','tsolver', CitcomModule)


# version
__id__ = "$Id: __init__.py,v 1.4 2003/07/24 17:46:47 tan2 Exp $"

# End of file
