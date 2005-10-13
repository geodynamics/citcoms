#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


def quadratic():
    from Quadratic import Quadratic
    return Quadratic()

def exponential():
    from Exponential import Exponential
    return Exponential()


def tabulator():
    from Tabulator import Tabulator
    return Tabulator()


def simple():
    from Simple import Simple
    return Simple()


def copyright():
    return "tabulator pyre module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

#  End of file 
