#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def coarsegridexchanger(name='cge', facility='cge'):
    from CoarseGridExchanger import CoarseGridExchanger
    return CoarseGridExchanger(name, facility)


def finegridexchanger(name='fge', facility='fge'):
    from FineGridExchanger import FineGridExchanger
    return FineGridExchanger(name, facility)



# version
__id__ = "$Id: __init__.py,v 1.4 2003/10/28 23:51:48 tan2 Exp $"

# End of file
