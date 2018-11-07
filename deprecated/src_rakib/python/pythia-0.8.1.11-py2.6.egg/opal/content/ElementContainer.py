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


class ElementContainer(Element):


    def add(self, item):
        self.contents.append(item)
        return self


    def __init__(self, tag, **attributes):
        Element.__init__(self, tag, **attributes)
        self.contents = []
        return


# version
__id__ = "$Id: ElementContainer.py,v 1.3 2005/03/27 10:17:16 aivazis Exp $"

# End of file 
