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


from Element import Element


class Title(Element):


    def identify(self, inspector):
        return inspector.onTitle(self)


    def __init__(self, title):
        Element.__init__(self, 'title')
        self.title = title
        return


# version
__id__ = "$Id: Title.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
