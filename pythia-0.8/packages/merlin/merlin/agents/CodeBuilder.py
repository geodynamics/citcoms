#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Builder import Builder


class CodeBuilder(Builder):


    def execute(self, merlin, project):
        if not project:
            return True
        
        languages = merlin.languages()
        for language in languages:
            handler = merlin.language(language)
            print "%s handler: %s" % (language, handler)

        return True


    def __init__(self):
        Builder.__init__(self)
        return


# version
__id__ = "$Id: CodeBuilder.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
