#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def coupler(name="coupler", facility="coupler"):
    from Coupler import Coupler
    return Coupler(name, facility)


# version
__id__ = "$Id: __init__.py,v 1.2 2003/09/05 19:49:15 tan2 Exp $"

# End of file
