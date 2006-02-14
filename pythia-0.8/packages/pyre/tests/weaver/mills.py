#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class Mills(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        name = pyre.inventory.str("name")
        language = pyre.inventory.str("language")


    def main(self, *args, **kwds):

        filename = self.filename
        weaver = self.weaver

        if filename:
            stream = file(filename, "w")
        else:
            import sys
            stream = sys.stdout

        document = []

        # pick languages to test
        if self.language:
            languages = [self.language]
        else:
            languages = weaver.languages()

        for language in languages:
            print language
            self.weaver.language = language
            document += self.weaver.render()

        text = "\n".join(document)
        print >> stream, text
        
        return


    def __init__(self):
        Script.__init__(self, "mills")
        self.filename = None
        self.language = None
        return


    def _init(self):
        Script._init(self)
        
        self.filename = self.inventory.name
        self.language = self.inventory.language
        return


# main

if __name__ == "__main__":
    app = Mills()
    app.run()


# version
__id__ = "$Id: mills.py,v 1.2 2005/03/13 19:41:25 aivazis Exp $"

# End of file
