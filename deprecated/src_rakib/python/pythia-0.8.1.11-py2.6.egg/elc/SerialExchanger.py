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


from Exchanger import Exchanger


class SerialExchanger(Exchanger):


    def exchange(self):
        return


    def __init__(self):
        Exchanger.__init__(self, "serial")
        return


# version
__id__ = "$Id: SerialExchanger.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

#  End of file 
