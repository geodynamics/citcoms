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


class Executive(object):


    # factories

    from CommandlineParser import CommandlineParser
    
    def createCommandlineParser(self):
        """create a command line parser"""

        return self.CommandlineParser()


    def createRegistry(self, name=None):
        """create a registry instance to store my configuration"""

        if name is None:
            name = self.name

        import pyre.inventory
        return pyre.inventory.registry(name)


    def createCurator(self, name=None):
        """create a curator to handle the persistent store"""

        if name is None:
            name = self.name

        import pyre.inventory
        curator = pyre.inventory.curator(name)

        return curator


    # configuration
    def getArgv(self, *args, **kwds):
        argv = kwds.get('argv')
        if argv is None:
            import sys
            argv = sys.argv
        self.arg0 = argv[0]
        self._requires = kwds.get('requires')
        argv = argv[1:]
        return argv


    def requires(self):
        # This was once used by Application.path() to construct a
        # "minimal" search path for the application.  Now it is only
        # used by version().
        if self._requires is None:
            from __main__ import __requires__
            self._requires = __requires__
        return self._requires


    def processCommandline(self, registry, argv=None, parser=None):
        """convert the command line arguments to a trait registry"""

        if parser is None:
            parser = self.createCommandlineParser()

        parser.parse(registry, argv)

        return parser


    def verifyConfiguration(self, context, mode='strict'):
        """verify that the user input did not contain any typos"""

        return context.verifyConfiguration(self, mode)


    # the default application action
    def main(self, *args, **kwds):
        return


    # user assistance
    def help(self):
        self.showHelp()
        return


    def complete(self):
        """perform shell command completion"""
        
        from glob import glob
        import os
        
        arg, prevArg = self.unprocessedArguments[1:3]
        if arg == "":
            line = os.environ['COMP_LINE']
            point = int(os.environ['COMP_POINT'])
            if line[point - 1] == "=":
                # NYI: value completion
                return
        
        parser = self.createCommandlineParser()
        prefix, fields, value, filenameStem = parser.parseArgument(arg, prevArg)

        # Match filenames.
        if filenameStem is not None:
            for codec in self.getCurator().codecs.itervalues():
                extension = "." + codec.extension
                if filenameStem:
                    pattern = "%s*" % filenameStem
                    for filename in glob(pattern):
                        if filename.endswith(extension):
                            print filename
                else:
                    pattern = "%s*%s" % (filenameStem, extension)
                    for filename in glob(pattern):
                        print filename

        # Match traits.
        if fields is not None:
            component = self
            path = ""
            for field in fields[:-1]:
                facilityNames = component.inventory.facilityNames()
                if not field in facilityNames:
                    return
                path += field + "."
                component = component.getTraitValue(field)
            if fields:
                field = fields[-1]
            else:
                field = ""
            propertyNames = component.inventory.propertyNames()
            candidates = []
            for prop in propertyNames:
                if prop.startswith(field):
                    candidates.append(prop)
            if not candidates:
                return
            if len(candidates) == 1:
                prop = candidates[0]
                facilityNames = component.inventory.facilityNames()
                if prop in facilityNames:
                    print "%s%s%s." % (prefix, path, prop)
                    print "%s%s%s=" % (prefix, path, prop)
                else:
                    print "%s%s%s" % (prefix, path, prop)
            else:
                for prop in candidates:
                    print "%s%s%s" % (prefix, path, prop)        

        return


    def usage(self):
        from os.path import basename
        print 'usage: %s [--<property>=<value>] [--<facility>.<property>=<value>] [FILE.cfg] ...' % basename(self.arg0)
        self.showUsage()
        return


    def version(self):
        from pkg_resources import get_provider, Requirement
        try:
            req = self.requires()
        except ImportError:
            print "Please consider writing version info for this application."
            return
        req = Requirement.parse(req)
        provider = get_provider(req)
        # NYI: make this pretty
        for line in provider.get_metadata_lines("PKG-INFO"):
            print line
        return


    def __init__(self):
        self.arg0 = self.name
        self._requires = None


# version
__id__ = "$Id: Executive.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
