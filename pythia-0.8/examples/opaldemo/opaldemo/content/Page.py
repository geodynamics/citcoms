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


from opal.content.Page import Page as Base


class Page(Base):


    def __init__(self, name, title, root):
        Base.__init__(self)

        head = self.head()
        head.base(url=root)
        head.title(title)

        head.stylesheet(rel="stylesheet", media="all", url="css/visual.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/visual-color.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/structural.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/structural-color.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/classes.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/classes-color.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/reports.css")
        head.stylesheet(rel="stylesheet", media="all", url="css/reports-color.css")

        self.home = '%s/%s.html' % (root, name)

        return


# version
__id__ = "$Id: Page.py,v 1.6 2005/04/27 18:01:23 pyre Exp $"

# End of file 
