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


def exchanger(name='exchanger', facility='exchanger'):
    from Exchanger import Exchanger
    return Exchanger

# version
__id__ = "$Id: __init__.py,v 1.5 2003/12/18 22:30:19 puru Exp $"

# End of file
