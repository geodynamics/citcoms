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

import os

if os.name == 'posix':
    from FileLockingPosix import FileLockingPosix as FileLocking

elif os.name == 'nt':
    from FileLockingNT import FileLockingNT as FileLocking

else:
    import journal
    warning = journal.warning('pyre.odb')
    warning.line("no file locking services are available for %s" % os.name)
    warning.log("please contact the pyre development team")


    class FileLocking(object):


        def lock(self, stream, flags):
            return


        def unlock(self, stream):
            return
    

# version
__id__ = "$Id: FileLocking.py,v 1.1.1.1 2005/03/08 16:13:41 aivazis Exp $"

# End of file 
