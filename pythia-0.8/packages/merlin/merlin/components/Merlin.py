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


from pyre.applications.Script import Script


class Merlin(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        actions = pyre.inventory.list('actions')


    # project definitions
    def spell(self):
        """retrieve the project definition file"""

        # NYI: embed the locator in the retrieved asset
        print " ** NYI"

        spell = self.retrieveComponent(
            name='', factory='project', args=[self], encoding='merlin')

        if spell:
            self._info.log("loaded project definition file from %s" % spell.getLocator())
        else:
            import journal
            journal.error("merlin").log("could not find project definition file")

        return spell
            

    def project(self, name, type):
        """load a project manager of type <type>"""

        project = self.retrieveComponent(
            name=type, factory='project', args=[name], vault=['projects'])

        if project:
            self._info.log("loaded project manager '%s' from %s" % (type, project.getLocator()))
        else:
            import journal
            journal.error("merlin").log("could not find project definition file for %r" % type)
            
        return project


    def agents(self, project, actions):
        """retrieve the agents for <actions> from the persistent store"""

        agents = []

        for action in actions:
            agent = self.retrieveComponent(
                name=action, factory='agent', args=[self, project], vault=['actions'])
            if agent:
                self._info.log("loaded action '%s' from %s" % (action, agent.getLocator()))
            else:
                import journal
                journal.warning('merlin').log("action '%s': no corresponding agent" % action)
                continue

            agents.append(agent)

        return agents


    def language(self, language):
        """retrieve <language> handler from the persistent store"""

        agent = self.retrieveComponent(
            name=language, factory='language', args=[self], vault=['languages'])

        if agent:
            self._info.log("loaded '%s' from %s" % (language, agent.getLocator()))
        else:
            import journal
            journal.warning('merlin').log("language '%s' not found" % language)

        return agent


    # indices
    def languages(self):
        return self.db.languages()


    # application interface
    def main(self, *args, **kwds):

        # load the project definition file
        project = self.spell()

        if project is None:
            self.improvise()
            return
            
        # load agents to take care of the requested actions
        actions = self.actions + self.argv
        if actions:
            self._info.log('actions: %s' % ", ".join(actions))
        else:
            self._info.log("no actions specified")

        agents = filter(None, self.agents(project, actions))
        for agent in agents:
            up, uc = self.configureComponent(agent)
            agent.init()
            agent.execute(self, project)
            agent.fini()

        return


    def improvise(self):
        import journal
        journal.warning("merlin").log("no valid project descriptions could be found")
        return True


    def createCurator(self, name=None):
        from Curator import Curator
        return Curator(name)


    def __init__(self):
        Script.__init__(self, 'merlin')
        self.db = None
        self.actions = []
        return


    def _init(self):
        self.db = self.inventory.getCurator()
        self.actions += self.inventory.actions
        return


# version
__id__ = "$Id: Merlin.py,v 1.2 2005/03/09 20:23:49 aivazis Exp $"

# End of file 
