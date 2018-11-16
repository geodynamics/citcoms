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


from pyre.parsing.locators.Traceable import Traceable


class Table(Traceable):


    def getValues(self):
        return [ self._priv_columns.get(name) for name in self._columnRegistry ]


    def getWriteableValues(self):
        return [ self._priv_columns.get(name) for name in self._writeable ]


    def getColumnNames(self):
        return self._columnRegistry.keys()


    def getWriteableColumnNames(self):
        return self._writeable


    def __init__(self):
        Traceable.__init__(self)
        
        # local storage for the descriptors created by the various traits
        self._priv_columns = {}

        return


    # the low level interface
    def _getColumnValue(self, name):
        return self._priv_columns[name]


    def _setColumnValue(self, name, value):
        self._priv_columns[name] = value
        return value


    # column registries
    _writeable = []
    _columnRegistry = {}


    # metaclass
    from Schemer import Schemer
    __metaclass__ = Schemer


# version
__id__ = "$Id: Table.py,v 1.4 2005/04/07 22:16:36 aivazis Exp $"

# End of file 
