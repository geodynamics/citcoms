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


from opal.applications.CGI import CGI
from pyre.applications.ComponentHarness import ComponentHarness


class WebApplication(CGI):


    class Inventory(CGI.Inventory):

        import pyre.inventory
        import opal.components
        import opal.inventory

        # properties
        home = pyre.inventory.str("home")
        home.meta['tip'] = "the path to my html files"

        cgihome = pyre.inventory.str("cgi-home")
        cgihome.meta['tip'] = "the url of the main application"

        routine = pyre.inventory.str("routine", default=None)
        routine.meta['tip'] = "the action to be performed by the actor"

        # components
        sentry = pyre.inventory.facility("sentry", factory=opal.components.sentry)
        sentry.meta['tip'] = "the ipa session manager"

        actor = opal.inventory.actor(default="error")
        actor.meta['tip'] = "the agent that defines the application behavior"


    def main(self, *args, **kwds):
        page = self.actor.perform(self, self.inventory.routine)
        self.render(page)
        return


    def retrieveActor(self, name):
        actor = self.retrieveComponent(name, factory='actor', vault=['actors'])
        return actor


    def retrieveForm(self, name):
        form = self.retrieveComponent(name, factory='form', vault=['forms'])
        return form


    def retrievePage(self, name):
        page = self.retrieveComponent(name, factory='page', args=[self], vault=['pages'])
        return page


    def retrievePortlet(self, name):
        page = self.retrieveComponent(name, factory='portlet', args=[self], vault=['portlets'])
        return page


    def render(self, page=None):
        self.weaver.renderer = self.pageMill
        self.weaver.weave(document=page, stream=self.stream)
        return


    def authenticate(self):
        return self.sentry.authenticate()


    def __init__(self, name, asCGI=None):
        CGI.__init__(self, name, asCGI)

        # our renderer
        self.pageMill = None

        # the authenticator
        self.sentry = None

        # the behavior
        self.actor = None
        self.routine = ""

        # the urls
        self.home = ''
        self.cgihome = ''

        return


    def _defaults(self):
        CGI._defaults(self)
        self.inventory.typos = 'relaxed'
        return


    def _configure(self):
        CGI._configure(self)

        # create our renderer
        import opal
        self.pageMill = opal.pageMill()

        # the authenticator
        self.sentry = self.inventory.sentry

        # the behavior
        self.actor = self.inventory.actor
        self.routine = self.inventory.routine

        # the home
        self.home = self.inventory.home

        # the cgi home
        self.cgihome = self.inventory.cgihome

        return


# version
__id__ = "$Id: WebApplication.py,v 1.9 2005/05/02 18:11:06 pyre Exp $"

# End of file 
