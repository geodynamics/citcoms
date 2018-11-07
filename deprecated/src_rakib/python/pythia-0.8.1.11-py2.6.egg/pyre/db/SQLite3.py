#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2010  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


import sqlite3
from DBManager import DBManager


class SQLite3(DBManager):


    # exceptions
    ProgrammingError = sqlite3.ProgrammingError


    # interface
    def connect(self, **kwds):
        return sqlite3.connect(**kwds)


    placeholder = "?"


# end of file 
