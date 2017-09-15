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


from Script import Script


class SuperScript(Script):


    def createCommandlineParser(self):
        import pyre.applications
        return pyre.applications.superCommandlineParser()


    def createSubscript(self, name):
        try:
            factory = self.subscripts[name]
        except KeyError:
            self._error.log("unknown command: %s" % name)
            self.usage()
            import sys
            sys.exit(1)
            
        subscript = factory()
        subscript.arg0 = name
        return subscript


    def execute(self, *args, **kwds):

        if len(self.argv) < 1:
            self.usage()
            import sys
            sys.exit(1)
        
        subscriptName = self.argv[-1]
        self.subscript = self.createSubscript(subscriptName)
        
        self.main(*args, **kwds)
        
        return


    def main(self, *args, **kwds):
        args = kwds.get('args', [])
        kwds = kwds.get('kwds', dict())
        kwds['argv'] = [self.argv[-1]] + self.unprocessedArguments
        self.runSubscript(*args, **kwds)


    def runSubscript(self, *args, **kwds):
        self.subscript.run(*args, **kwds)


    def usage(self):
        print 'usage: %s [options] <command> [COMMAND-ARG...]' % self.arg0
        return


# end of file
