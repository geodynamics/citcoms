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


def coarsegridexchanger(name, facility='exchanger'):
    from CoarseGridExchanger import CoarseGridExchanger
    return CoarseGridExchanger(name, facility)


def finegridexchanger(name, facility='exchanger'):
    from FineGridExchanger import FineGridExchanger
    return FineGridExchanger(name, facility)



# version
__id__ = "$Id: __init__.py,v 1.2 2003/09/05 19:49:14 tan2 Exp $"

# End of file
