#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Renderer(object):


    def render(self, entry, device):

        text = []
        meta = entry.meta

        if self.header:
            filename = meta["filename"]
            if self.trimFilename and len(filename) > 53:
                filename = filename[0:20] + "..." + filename[-30:]
                meta["filename"] = filename
            text.append(self.header % self.subst(meta, device))

        for line in entry.text:
            text.append(self.format % line)

        if self.footer:
            text.append(self.footer % self.subst(meta, device))

        return text


    def subst(self, dct, device):
        return dct


    def __init__(self, header=None, format=None, footer=None):
        if header is None:
            header = " >> %(filename)s:%(line)s:%(function)s\n >> %(facility)s(%(severity)s)"

        if format is None:
            format = " -- %s"

        if footer is None:
            footer = ""

        self.header = header
        self.format = format
        self.footer = footer

        self.trimFilename = False

        return


# version
__id__ = "$Id: Renderer.py,v 1.1.1.1 2005/03/08 16:13:53 aivazis Exp $"

#  End of file 
