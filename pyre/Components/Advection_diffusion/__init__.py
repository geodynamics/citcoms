#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def temperature_diffadv(name, facility='tsolver'):
    from Advection_diffusion import Advection_diffusion
    return Advection_diffusion(name, facility)

    #from Temperature_diffadv import Temperature_diffadv
    #return Temperature_diffadv('temp','tsolver', CitcomModule)


# version
__id__ = "$Id: __init__.py,v 1.6 2003/08/27 20:52:47 tan2 Exp $"

# End of file
