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


class Sentry(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        username = pyre.inventory.str('username')
        username.meta['tip'] = "the requestor's username"

        passwd = pyre.inventory.str('passwd')
        passwd.meta['tip'] = "the requestor's passwd"

        ticket = pyre.inventory.str('ticket')
        ticket.meta['tip'] = "the requestor's previously obtained ticket"

        attempts = pyre.inventory.int('attempts')
        attempts.meta['tip'] = "the number of unsuccessful attempts to login"

        import pyre.ipa
        ipa = pyre.inventory .facility("session", family="ipa", factory=pyre.ipa.session)
        ipa.meta['tip'] = "the ipa session manager"


    def authenticate(self):
        self.attempts += 1
        if self.ticket:
            try:
                self.ticket = self.ipa.refresh(self.username, self.ticket)
                return self.ticket
            except self.ipa.RequestError:
                return

        try:
            self.ticket = self.ipa.login(self.username, self.passwd)
            return self.ticket
        except self.ipa.RequestError:
            return

        return


    def __init__(self, name=None):
        if name is None:
            name = 'sentry'

        Component.__init__(self, name, facility='sentry')

        # the user parameters
        self.username = ''
        self.passwd = ''
        self.ticket = ''
        self.attempts = 0

        # the IPA session
        self.ipa = None

        return


    def _configure(self):
        Component._configure(self)
        self.username = self.inventory.username
        self.passwd = self.inventory.passwd
        self.ticket = self.inventory.ticket
        self.attempts = self.inventory.attempts

        self.ipa = self.inventory.ipa

        return


# version
__id__ = "$Id: Sentry.py,v 1.5 2005/04/14 08:52:43 aivazis Exp $"

# End of file 
