#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.parsing.locators.Traceable import Traceable


class Person(Traceable):


    def __init__(self):
        Traceable.__init__(self)
        
        self.id = ''
        self.first = ''
        self.middle = ''
        self.last = ''
        return


    def __str__(self):
        return " ".join(filter(None, [self.first, self.middle, self.last]))


# version
__id__ = "$Id: Person.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
