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

try:
    stem = os.environ["PYTHIA_HOME"]
except KeyError:
    try:
        stem = os.environ["EXPORT_ROOT"]
    except KeyError:
        stem = 'xxDBROOTxx'

_SYSTEM_ROOT = os.path.join(stem, "etc")
_USER_ROOT = os.path.join(os.path.expanduser('~'), '.pyre')
_LOCAL_ROOT = [ '.' ]


# version
__id__ = "$Id: prefix-template.py,v 1.1.1.1 2005/03/08 16:13:43 aivazis Exp $"

# End of file 
