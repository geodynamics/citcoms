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


class Base(Element):


    def identify(self, inspector):
        return inspector.onBase(self)


    def __init__(self, url):
        Element.__init__(self, 'base', href=url)
        return


# version
__id__ = "$Id: Base.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
