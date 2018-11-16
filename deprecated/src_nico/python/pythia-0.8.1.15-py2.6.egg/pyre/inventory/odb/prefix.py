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

import os
from os.path import dirname
from pyre import __version__

_SYSTEM_ROOT = '/etc/pythia-' + __version__ # PORTABILITY: unix only
_USER_ROOT = os.path.join(os.path.expanduser('~'), '.pyre')
_LOCAL_ROOT = [ '.' ]


# version
__id__ = "$Id: prefix-template.py,v 1.1.1.1 2005/03/08 16:13:43 aivazis Exp $"

# End of file 
