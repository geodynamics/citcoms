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


def run():
    import pyre.idd

    app = pyre.idd.daemon()
    return app.run(spawn=True)
    

if __name__ == "__main__":
    run()


# version
__id__ = "$Id: idd.py,v 1.2 2005/03/09 07:04:15 aivazis Exp $"

# End of file 
