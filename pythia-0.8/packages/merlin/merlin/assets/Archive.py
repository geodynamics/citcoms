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

from Asset import Asset


class Archive(Asset):


    def identify(self, inspector):
        return inspector.onArchive(self)


    def __init__(self, name):
        Asset.__init__(self, name)

        from Sources import Sources
        self._sources = Sources(self.name)
        return


    def _getSources(self):
        return self._sources


    def _setSources(self, files):
        from Source import Source
        sources = [ Source(name) for name in files ]
        self._sources.append(sources)
        return


    sources = property(_getSources, _setSources, None, "")


# version
__id__ = "$Id: Archive.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
