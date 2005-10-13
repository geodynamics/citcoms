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


class TagMill(object):


    def onElement(self, element):
        text = [ "%s" % element.tag ]
        text += self.renderAttributes(element.attributes)
        return "<%s/>" % " ".join(text)


    def onElementBegin(self, element):
        text = [ "%s" % element.tag ]
        text += self.renderAttributes(element.attributes)
        return "<%s>" % " ".join(text)


    def onElementEnd(self, element):
        return "</%s>" % element.tag


    def onCoreAttributes(self, attributes):
        text = []
        if attributes.cls:
            text.append('class="%s"' % attributes.cls)

        if attributes.id:
            text.append('id="%s"' % attributes.id)

        if attributes.style:
            text.append('title="%s"' % attributes.title)

        if attributes.title:
            text.append('title="%s"' % attributes.title)

        return " ".join(text)


    def onLanguageAttributes(self, attributes):
        if attributes.dir:
            text.append('dir="%s"' % attributes.dir)

        if attributes.lang:
            text.append('lang="%s"' % attributes.lang)

        return " ".join(text)


    def onKeyboardAttributes(self, attributes):
        if attributes.accesskey:
            text.append('accesskey="%s"' % attributes.accesskey)

        if attributes.tabindex:
            text.append('tabindex="%s"' % attributes.tabindex)

        return " ".join(text)


    def renderAttributes(self, attributes):
        try:
            attributes['class'] = attributes['cls']
            del attributes['cls']
        except KeyError:
            pass
        
        return [ '%s=%r' % (key, str(value)) for key, value in attributes.iteritems() ]


# version
__id__ = "$Id: TagMill.py,v 1.3 2005/05/03 03:00:41 pyre Exp $"

# End of file 
