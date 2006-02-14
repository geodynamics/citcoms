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


from pyre.odb.fs.CodecODB import CodecODB


class CodecPML(CodecODB):


    def __init__(self):
        CodecODB.__init__(self, encoding='pml')

        from Parser import Parser
        self._parser = Parser()

        self.parserFactory = None
        
        return


    def _createRenderer(self):
        from Renderer import Renderer
        return Renderer()


    def _decode(self, shelf):
        """lock and then read the contents of the file into the shelf"""

        stream = file(shelf.name)

        self._locker.lock(stream, self._locker.LOCK_EX)
        document = self._parser.parse(stream, self.parserFactory)
        self._locker.unlock(stream)

        shelf['blade'] = document
        shelf._frozen = True
        
        return


# version
__id__ = "$Id: CodecPML.py,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $"

# End of file 
