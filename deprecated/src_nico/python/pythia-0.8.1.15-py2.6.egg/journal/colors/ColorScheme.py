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


from pyre.components import Component


class ColorScheme(Component):


    import pyre.inventory as pyre

    filename         = pyre.str("filename",          default="NoColor")
    line             = pyre.str("line",              default="NoColor")
    function         = pyre.str("function",          default="NoColor")
    stackTrace       = pyre.str("stack-trace",       default="NoColor")

    src              = pyre.str("src",               default="NoColor")

    facility         = pyre.str("facility",          default="NoColor")
    severityDebug    = pyre.str("severity-debug",    default="NoColor")
    severityInfo     = pyre.str("severity-info",     default="NoColor")
    severityWarning  = pyre.str("severity-warning",  default="NoColor")
    severityError    = pyre.str("severity-error",    default="NoColor")
    
    normal           = pyre.str("normal",            default="Normal")


    def __getitem__(self, key):
        return self.getTraitValue(key)

    
# end of file 
