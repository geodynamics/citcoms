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


from ElementContainer import ElementContainer
from pyre.parsing.locators.Traceable import Traceable


class Portlet(ElementContainer, Traceable):


    def identify(self, inspector):
        return inspector.onPortlet(self)


    def form(self, **kwds):
        from PortletContent import PortletContent
        wrapper = PortletContent()

        from Form import Form
        item = Form(**kwds)
        wrapper.content = item

        self.contents.append(wrapper)
        return item


    def item(self, **kwds):
        from PortletContent import PortletContent
        wrapper = PortletContent()

        from PortletLink import PortletLink
        item = PortletLink(**kwds)
        wrapper.content = item

        self.contents.append(wrapper)
        return item


    def __init__(self, title, cls="portlet", **kwds):
        ElementContainer.__init__(self, tag='div', cls=cls, **kwds)
        Traceable.__init__(self)
        self.title = title
        return


# version
__id__ = "$Id: Portlet.py,v 1.7 2005/05/05 04:44:48 pyre Exp $"

# End of file 
