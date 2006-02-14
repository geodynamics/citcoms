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


from DocumentMill import DocumentMill


class StructuralMill(DocumentMill):


    def onDocument(self, document):
        text = [
            self.tagger.onElementBegin(document),
            '<h1>%s</h1>' % document.title,
            '<div class="documentDescription">',
            document.description,
            '</div>',
            ]

        for item in document.contents:
            text += item.identify(self)

        text += [
            '<div class="documentByline">',
            document.byline,
            '</div>',
            self.tagger.onElementEnd(document),
            ]

        return text


    def onLogo(self, logo):
        text = [
            '<div id="page-logo"><a href="%s">Home</a></div>' % logo.href
            ]
        
        return text


    def onPersonalTools(self, tools):
        text = [
            '<p id="page-personaltools">&nbsp;</p>'
            ]
        return text


    def onPageSection(self, section):
        text = [
            '<!-- ** section %s ** -->' % section.attributes.get('id', ''),
            self.tagger.onElementBegin(section),
            ]

        for tag in section.contents:
            text += tag.identify(self)
            
        text += [
            self.tagger.onElementEnd(section),
            '<!-- ** end of section %s ** -->' % section.attributes.get('id', '')
            ]
        
        return text


    def onPortlet(self, portlet):
        text = [
            '<div class="visualPadding">',
            self.tagger.onElementBegin(portlet),
            '    <h5>%s</h5>' % portlet.title,
            '    <div class="portletBody">',
            ]

        for item in portlet.contents:
            text += item.identify(self)

        text += [
            '    </div>',
            self.tagger.onElementEnd(portlet),
            '</div>',
            ]
        
        return text


    def onPortletContent(self, item):
        text = [
            self.tagger.onElementBegin(item)
            ]

        if item.content:
            text += item.content.identify(self)

        text.append( self.tagger.onElementEnd(item))

        return text


    def onPortletLink(self, link):
        if link.icon is None:
            icon = ""
        else:
            icon = '<img class="%sIcon" height="16" width="16" src="%s" />' % (
                link.type, link.icon)

        text = [
            '<a class="%s" title="%s"' % (link.type, link.tip),
            '  href="%s">' % link.target,
            icon,
            '<span class="%sText">%s</span>' % (link.type, link.description),
            '</a>',
            ]

        return text


    def onSearchBox(self, box):
        text = [
            '<!-- ** search box ** -->',
            '<div id="page-searchBox">',
            '  <form name="searchForm" action=" ">',
            '    <label for="searchGadget" class="hiddenObject">Search</label>',
            '    <input id="searchGadget" name="SearchableText" type="text"',
            '      size="20" alt="Search" title="Search" accesskey="s"',
            '      class="visibility:visible" tabindex="30001" />',
            '    <input class="searchButton" type="submit" value="Search"',
            '      accesskey="s" tabindex="30002" />',
            '  </form>',
            '</div>',
            '<!-- ** end of search box ** -->',
            ]

        return text


    def __init__(self, tagger):
        DocumentMill.__init__(self, tagger)
        return


# version
__id__ = "$Id: StructuralMill.py,v 1.9 2005/05/05 04:46:32 pyre Exp $"

# End of file 
