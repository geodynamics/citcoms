#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.odb.fs.CodecODB import CodecODB


class CodecTmpl(CodecODB):

    def __init__(self):
        CodecODB.__init__(self, encoding='tmpl')
        return

    def _decode(self, shelf):
        from Cheetah.Template import Template
        template = Template(file=shelf.name)
        shelf['template'] = template
        shelf._frozen = True
        return


# end of file
