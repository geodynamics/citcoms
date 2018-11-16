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


class ScriptLocator(object):


    def __init__(self, source, line, function):
        self.source = source
        self.line = line
        self.function = function
        return


    def __str__(self):
        return "{file=%r, line=%r, function=%r}" % (self.source, self.line, self.function)


    def __getstate__(self):
        return dict(source = self.source, line = self.line, function = self.function)


    def __setstate__(self, dict):
        self.source   = dict['source']
        self.line     = dict['line']
        self.function = dict['function']
        return


    def getAttributes(self, attr):
        import linecache
        attr["filename"] = self.source
        attr["line"] = self.line
        attr["function"] = self.function
        attr["src"] = linecache.getline(self.source, self.line).rstrip()
        return


    __slots__ = ("source", "line", "function")

# version
__id__ = "$Id: ScriptLocator.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
