#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from math import pi
from SI import radian

degree = pi/180 * radian
arcminute = degree / 60
arcsecond = arcminute / 60

deg = degree
rad = radian

# version
__id__ = "$Id: angle.py,v 1.2 2005/03/19 20:45:27 aivazis Exp $"

#
# End of file
