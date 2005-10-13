#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

if __name__ == "__main__":

    import mpi
    from mpi import _mpi as mpimodule

    print "copyright information:"
    print "   ", mpi.copyright()
    print "   ", mpimodule.copyright()

    print
    print "module information:"
    if mpimodule.initialized:
        print "    file:", mpimodule.__file__
    print "    doc:", mpimodule.__doc__
    print "    contents:", dir(mpimodule)

# version
__id__ = "$Id: signon.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

#  End of file 
