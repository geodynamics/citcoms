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


from ContentMill import ContentMill


class BodyMill(ContentMill):


    def onLiteral(self, literal):
        return [ "\n".join(literal.text) ]


    def onParagraph(self, p):
        text = [ self.tagger.onElementBegin(p) ]
        text += p.text
        text.append(self.tagger.onElementEnd(p))
        return text


    def onPageSection(self, section):
        return self.structuralMill.onPageSection(section)


    def onPageContent(self, page):
        text = [
            '<div class="visualClear"></div>',
            '<table id="page-content">',
            '  <tbody>',
            '    <tr>',
            ]

        if page._leftColumn:
            text += [
                '      <td id="page-columnLeft">',
                #'        <div class="visualPadding">'
                ]
            text += page._leftColumn.identify(self.structuralMill)
            text += [
                #'        </div>',
                '      </td>',
                ]

        if page._main:
            text += [
                '      <td id="page-main">',
                '        <div class="visualPadding">',
                ]
            text += page._main.identify(self.structuralMill)
            text += [
                '        </div>',
                '      </td>',
                ]

        if page._rightColumn:
            text += [
                '      <td id="page-columnRight">',
                #'        <div class="visualPadding">',
                ]
            text += page._rightColumn.identify(self.structuralMill)
            text += [
                #'        </div>',
                '      </td>',
                ]

        text += [
            '    </tr>',
            '  </tbody>',
            '</table>',
            ]

        return text


    def render(self, body):
        text = [
            self.tagger.onElementBegin(body),
            '<div id="body-wrapper">'
            ]

        for tag in body.contents:
            text += tag.identify(self)
            
        text += [
            '</div>',
            self.tagger.onElementEnd(body)
            ]
        
        return text


    def __init__(self, tagger):
        ContentMill.__init__(self)
        
        self.tagger = tagger

        from StructuralMill import StructuralMill
        self.structuralMill = StructuralMill(tagger)

        return


# version
__id__ = "$Id: BodyMill.py,v 1.4 2005/03/26 01:51:29 aivazis Exp $"

# End of file 
