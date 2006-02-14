#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                       California Institute of Technology
#                       (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from AssetContainer import AssetContainer


class Library(AssetContainer):


    def identify(self, inspector):
        return inspector.onLibrary(self)


    def __init__(self, name):
        AssetContainer.__init__(self, name)

        from Archive import Archive
        self.archive = Archive(name)

        from Headers import Headers
        self._headers = Headers(name)

        return


    def _getHeaders(self):
        return self._headers


    def _setHeaders(self, headers):
        from Header import Header
        headers = [ Header(name) for name in headers ]
        self._headers.append(headers)
        return


    def _getSources(self):
        return self.archive.sources


    def _setSources(self, files):
        self.archive.sources = files
        return


    headers = property(_getHeaders, _setHeaders, None, "")
    sources = property(_getSources, _setSources, None, "")


# version
__id__ = "$Id: Library.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
