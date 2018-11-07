#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2007  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class StackTraceLocator(object):


    def __init__(self, stackTrace):
        self.stackTrace = stackTrace
        return


    def __str__(self):
        filename, line, function, src = self.stackTrace[-1]
        return "{file=%r, line=%r, function=%r}" % (filename, line, function)


    def getAttributes(self, attr):
        filename, line, function, src = self.stackTrace[-1]
        attr["stack-trace"] = self.stackTrace
        attr["filename"] = filename
        attr["function"] = function
        attr["line"] = line
        attr["src"] = src
        return


# end of file 
