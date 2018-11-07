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

class RequestError(Exception):


    def __init__(self, msg):
        self.msg = msg
        return


    def __str__(self):
        return self.msg


# version
__id__ = "$Id: RequestError.py,v 1.1 2005/03/14 23:00:21 aivazis Exp $"

# End of file 
