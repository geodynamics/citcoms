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


from Device import Device


class Stream(Device):


    def createDevice(self):
        from journal.devices.File import File
        return File(self.stream)


    def __init__(self, stream, name=None):
        if name is None:
            name = "stream"
        Device.__init__(self, name)
        self.stream = stream
        return


# end of file 
