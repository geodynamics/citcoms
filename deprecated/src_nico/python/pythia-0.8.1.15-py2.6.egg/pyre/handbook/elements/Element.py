#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class Element(object):


    def __init__(self, number, symbol, name, weight):
        self.name = name
        self.symbol = symbol
        self.atomicNumber = number
        self.atomicWeight = weight
        return


    def __str__(self):
        return "%s (%s) - atomic number: %d, atomic weight: %g amu" \
               % (self.name, self.symbol, self.atomicNumber, self.atomicWeight)


# version
__id__ = "$Id: Element.py,v 1.1.1.1 2005/03/08 16:13:44 aivazis Exp $"

#
# End of file
