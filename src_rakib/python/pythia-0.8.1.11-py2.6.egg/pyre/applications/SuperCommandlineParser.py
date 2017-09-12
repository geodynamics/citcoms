#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from CommandlineParser import CommandlineParser


class SuperCommandlineParser(CommandlineParser):


    def _filterNonOptionArgument(self, arg):
        candidate = super(SuperCommandlineParser, self)._filterNonOptionArgument(arg)
        if candidate:
            return candidate
        # 'arg' is the first non-option argument -- we are done
        self.unprocessed.extend(self.argv)
        self.argv = []
        return None


# end of file
