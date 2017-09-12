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


class HeadMill(object):


    def render(self, head):
        text = [ self.tagger.onElementBegin(head) ]
        for tag in head.contents:
            text.append(tag.identify(self))
        text.append(self.tagger.onElementEnd(head))
        return text


    def onBase(self, base):
        return self.tagger.onElement(base)


    def onIncludedStyle(self, style):
        return "%s%s%s" % (
            self.tagger.onElementBegin(style),
            "@import url(%s);" % style.url,
            self.tagger.onElementEnd(style))


    def onLink(self, link):
        return self.tagger.onElement(link)


    def onMeta(self, meta):
        return self.tagger.onElement(meta)


    def onScript(self, script):
        return "%s\n%s\n%s" % (
            self.tagger.onElementBegin(script),
            "\n".join(script.script),
            self.tagger.onElementEnd(script))


    def onStyle(self, style):
        return "%s\n%s\n%s" % (
            self.tagger.onElementBegin(style),
            "\n".join(style.style),
            self.tagger.onElementEnd(style))


    def onTitle(self, title):
        return "%s%s%s" % (
            self.tagger.onElementBegin(title), title.title, self.tagger.onElementEnd(title))


    def __init__(self, tagger):
        self.tagger = tagger
        return


# version
__id__ = "$Id: HeadMill.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
