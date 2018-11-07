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

import journal
from Index import Index


class Info(Index):


    def init(self):
        Index.init(self, "info", False)
        return


    def _proxyState(self, name):
        from ProxyState import ProxyState
        return ProxyState(journal._journal.info(name))


# version
__id__ = "$Id: Info.py,v 1.1.1.1 2005/03/08 16:13:53 aivazis Exp $"

#  End of file 
