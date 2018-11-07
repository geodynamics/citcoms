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


class DocumentMill(ContentMill):


    def onForm(self, form):
        if form.legend:
            legendStart = '<fieldset><legend>%s</legend>' % form.legend
            legendEnd = '</fieldset>'
        else:
            legendStart = ''
            legendEnd = ''
            
        text = [
            self.tagger.onElementBegin(form),
            legendStart,
            ]

        for item in form.contents:
            text += item.identify(self)

        text += [
            legendEnd,
            self.tagger.onElementEnd(form),
            ]

        return text


    def onFormControl(self, control):
        text = [
            self.tagger.onElementBegin(control),
            '<input class="context" name="%s" type="%s" value="%s"/>' % (
                control.name, control.type, control.value),
            self.tagger.onElementEnd(control),
            ]
        
        return text


    def onFormField(self, field):
        control = field.control
        if not control:
            return []
        
        if control.label:
            label = '<label for="%s">%s</label>' % (control.attributes["id"], control.label)
        else:
            label = ''

        if control.help:
            help = '<div class="formfieldHelp">%s</div>' % control.help
        else:
            help = ''

        text = [
            self.tagger.onElementBegin(field),
            label,
            help
            ]

        text += control.identify(self)
        
        text += [
            self.tagger.onElementEnd(field),
            ]

        return text


    def onFormHiddenInput(self, control):
        text = [
            self.tagger.onElement(control)
            ]

        return text


    def onInput(self, control):
        text = [
            self.tagger.onElement(control)
            ]

        return text


    def onSelector(self, selector):

        text = [
            self.tagger.onElementBegin(selector)
            ]

        for value, description in selector.entries:
            if value == selector.selection:
                selected = "selected"
            else:
                selected = ""
            option = '  <option %s value="%s">%s' % (selected, value, description)
            text.append(option)

        text += [
            self.tagger.onElementEnd(selector),
            ]

        return text


    def __init__(self, tagger):
        ContentMill.__init__(self)
        self.tagger = tagger
        return

# version
__id__ = "$Id: DocumentMill.py,v 1.5 2005/05/05 17:44:22 pyre Exp $"

# End of file 
