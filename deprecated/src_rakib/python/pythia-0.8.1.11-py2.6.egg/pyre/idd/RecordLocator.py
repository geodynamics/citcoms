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


from pyre.components.Component import Component


class RecordLocator(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        alphabet = pyre.inventory.str('alphabet', default="23456789ABCDEFGHIJKLMNPQRSTUVWXYZ")


    def decode(self, locator):
        locator = locator.upper()
        locator = list(locator)
        locator.reverse()

        tid = 0
        for index, letter in enumerate(locator):
            tid += self._hashtable[letter] * self._base**index

        label = str(tid)

        return label[:-4], label[-4:]
            
        

    def encode(self, transactionId, date=None):
        if date is None:
            import time
            tick = time.localtime()
            date = time.strftime("%y%m%d", tick)

        bcd = int(str(transactionId) + date)

        locator = self._encode(bcd)

        return locator


    def __init__(self):
        Component.__init__(self, "locator", facility="recordLocator")
        self._alphabet = None
        self._base = None
        self._hashtable = None

        return


    def _init(self):
        Component._init(self)
        
        self._alphabet = list(self.inventory.alphabet)
        self._base = len(self._alphabet)
        self._hashtable = self._hash(self._alphabet)

        return


    def _encode(self, bcd):
        label = []

        while 1:
            bcd, remainder = divmod(bcd, self._base)
            label.append(self._alphabet[remainder])

            if bcd == 0:
                break

        label.reverse()
        label = "".join(label)
        return label
        

    def _hash(self, alphabet):
        hash = {}

        for index, letter in enumerate(alphabet):
            hash[letter] = index

        return hash


# version
__id__ = "$Id: RecordLocator.py,v 1.3 2005/04/28 03:37:16 pyre Exp $"

# End of file 
