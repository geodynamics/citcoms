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


class Source(Asset):


    def identify(self, inspector):
        return inspector.onSource(self)


    def __init__(self, file, language=None):
        Asset.__init__(self, file)

        self.language = None #self._inferLanguage(file, language)

        return


    # helpers
    
    def _inferLanguage(filename, guess):
        import merlin.languages
        language = merlin.languages.deduce(filename, guess)
        return language


    _inferLanguage = staticmethod(_inferLanguage)


# version
__id__ = "$Id: Source.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
