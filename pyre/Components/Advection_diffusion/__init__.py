#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def temperature_diffadv(name='temp', facility='tsolver'):
    from Advection_diffusion import Advection_diffusion
    return Advection_diffusion(name, facility)


# version
__id__ = "$Id: __init__.py,v 1.7 2003/10/28 23:51:48 tan2 Exp $"

# End of file
