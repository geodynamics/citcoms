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


import psycopg
from DBManager import DBManager


class Psycopg(DBManager):


    # exceptions
    ProgrammingError = psycopg.ProgrammingError


    # interface
    def connect(self, **kwds):
        return psycopg.connect(**kwds)


# version
__id__ = "$Id: Psycopg.py,v 1.2 2005/04/05 23:31:15 aivazis Exp $"

# End of file 
