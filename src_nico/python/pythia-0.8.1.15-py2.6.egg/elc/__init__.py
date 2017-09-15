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

# exchanger factories
def iceExchanger():
    from ICEExchanger import ICEExchanger
    return ICEExchanger()


def mpiExchanger():
    from MPIExchanger import MPIExchanger
    return MPIExchanger()


def serialExchanger():
    from SerialExchanger import SerialExchanger
    return SerialExchanger()


# copyright note
def copyright():
    return "elc pyre module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

#  End of file 
