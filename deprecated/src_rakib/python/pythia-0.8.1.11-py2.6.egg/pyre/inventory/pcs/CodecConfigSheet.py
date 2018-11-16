#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.inventory.odb.Registry import Registry
from pyre.odb.fs.CodecODB import CodecODB
from Parser import Parser


class CodecConfigSheet(CodecODB):

    def __init__(self):
        CodecODB.__init__(self, encoding='pcs')
        return

    def _decode(self, shelf):
        root = Registry("root")
        parser = Parser(root)
        parser.parseFile(shelf.name)
        shelf['inventory'] = root
        shelf._frozen = True
        return


# end of file
