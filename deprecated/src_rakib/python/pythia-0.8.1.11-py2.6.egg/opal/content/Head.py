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


class Head(ElementContainer):


    def base(self, url):
        from Base import Base
        base = Base(url)
        self.contents.append(base)
        return base


    def stylesheet(self, **kwds):
        from IncludedStyle import IncludedStyle
        style = IncludedStyle(**kwds)
        self.contents.append(style)
        return style


    def link(self, **kwds):
        from Link import Link
        link = Link(**kwds)
        self.contents.append(link)
        return link


    def meta(self, name, content):
        from Meta import Meta
        meta = Meta(name=name, content=content)
        self.contents.append(meta)
        return meta


    def script(self, **kwds):
        from Script import Script
        script = Script(**kwds)
        self.contents.append(script)
        return script


    def style(self, **kwds):
        from Style import Style
        style = Style(**kwds)
        self.contents.append(style)
        return style


    def title(self, text):
        from Title import Title
        title = Title(text)
        self.contents.append(title)
        return title


    def identify(self, inspector):
        return inspector.onHead(self)


    def __init__(self, **kwds):
        ElementContainer.__init__(self, 'head', **kwds)
        return


# version
__id__ = "$Id: Head.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
