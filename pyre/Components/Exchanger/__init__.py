#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



def exchanger(name, facility='exchanger'):
    from Exchanger import Exchanger
    return Exchanger(name, facility)


# version
__id__ = "$Id: __init__.py,v 1.1 2003/08/30 00:44:19 tan2 Exp $"

# End of file
