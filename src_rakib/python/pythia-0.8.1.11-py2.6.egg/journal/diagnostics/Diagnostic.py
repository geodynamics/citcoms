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
from pyre.parsing.locators import here
from Entry import Entry


class Diagnostic(object):


    def line(self, message):
        if not self.state:
            return

        self._entry.line(message)
        return self


    def log(self, message=None, locator=None):
        if not self.state:
            return

        if message is not None:
            self._entry.line(message)

        meta = self._entry.meta
        
        if locator is None:
            locator = here(1)

        # These are required by 'journal.devices.Renderer'.
        meta["filename"] = ""
        meta["function"] = ""
        meta["line"] = ""
        
        locator.getAttributes(meta)

        meta["facility"] = self.facility
        meta["severity"] = self.severity

        journal.journal().record(self._entry)

        if self.fatal:
            raise self.Fatal(message)
     
        self._entry = Entry()
        return self


    def activate(self):
        self._state.set(True)
        return self


    def deactivate(self):
        self._state.set(False)
        return self

    def flip(self):
        self._state.flip()
        return self


    def __init__(self, facility, severity, state, fatal=False):
        self.facility = facility
        self.severity = severity
        
        self._entry = Entry()
        self._state = state
        self.fatal = fatal

        return


    def _getState(self):
        return self._state.get()
    

    def _setState(self, state):
        self._state.set(state)
        return
    

    state = property(_getState, _setState, None, "")


    class Fatal(SystemExit):


        def __init__(self, msg=""):
            self.msg = msg


        def __str__(self):
            return self.msg

    
    # C++ interface

    def attribute(self, key, value):
        meta = self._entry.meta
        meta[key] = value


    def record(self):
        if not self.state:
            return

        meta = self._entry.meta
        meta["facility"] = self.facility
        meta["severity"] = self.severity
        meta.setdefault("filename", "<unknown>")
        meta.setdefault("function", "<unknown>")
        meta.setdefault("line", "<unknown>")

        journal.journal().record(self._entry)

        if self.fatal:
            raise self.Fatal()
     
        self._entry = Entry()
        return self


# version
__id__ = "$Id: Diagnostic.py,v 1.1.1.1 2005/03/08 16:13:53 aivazis Exp $"

#  End of file 
