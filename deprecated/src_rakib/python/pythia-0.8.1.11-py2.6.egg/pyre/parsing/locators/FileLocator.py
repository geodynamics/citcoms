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


class FileLocator(object):


    def __init__(self, source, line, column):
        self.source = source
        self.line = line
        self.column = column
        return


    def __str__(self):
        return "{file=%r, line=%r, column=%r}" % (self.source, self.line, self.column)


    def __getstate__(self):
        return dict(source = self.source, line = self.line, column = self.column)


    def __setstate__(self, dict):
        self.source = dict['source']
        self.line   = dict['line']
        self.column = dict['column']
        return


    def getAttributes(self, attr):
        import linecache
        attr["filename"] = self.source
        attr["line"] = self.line
        attr["src"] = linecache.getline(self.source, self.line).rstrip()
        return
    

    __slots__ = ("source", "line", "column")


# version
__id__ = "$Id: FileLocator.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
