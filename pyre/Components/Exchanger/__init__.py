#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def coarsegridexchanger(name='cge', facility='exchanger'):
    from CoarseGridExchanger import CoarseGridExchanger
    return CoarseGridExchanger(name, facility)


def finegridexchanger(name='fge', facility='exchanger'):
    from FineGridExchanger import FineGridExchanger
    return FineGridExchanger(name, facility)



# version
__id__ = "$Id: __init__.py,v 1.3 2003/10/24 05:04:31 tan2 Exp $"

# End of file
