#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def blade():
    """create the UI manager"""

    from components.Blade import Blade
    return Blade()


# misc
def copyright():
    return "blade: Copyright (c) 1998-2005 Michael A.G. Aivazis"

# version
__version__ = "0.8"
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $"

# End of file 
