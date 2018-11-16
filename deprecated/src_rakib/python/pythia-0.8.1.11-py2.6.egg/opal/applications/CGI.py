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


class CGI(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        stream = pyre.inventory.outputFile("stream")
        stream.meta['tip'] = "where to place the generated text"


    def collectUserInput(self, registry, context):

        # look through unprocessed arguments for GET input
        self.collectCGIInput(registry, self.unprocessedArguments)

        return


    def collectCGIInput(self, registry, argv):
        # get access to the environment variables
        import os
        
        # create a parser for query strings
        parser = self._createCGIParser()

        # figure out the request method
        try:
            method = os.environ['REQUEST_METHOD'].upper()
        except KeyError:
            method = 'GET'

        # extract the headers
        headers = {}
        headers['content-type'] = os.environ.get(
            'CONTENT_TYPE', 'application/x-www-form-urlencoded')
        try:
            headers['content-length'] = os.environ['CONTENT_LENGTH']
        except KeyError:
            pass

        # process arguments from query string
        if method == 'GET' or method == 'HEAD':
            try:
                query = os.environ['QUERY_STRING']
            except KeyError:
                pass
            else:
                parser.parse(registry, query, 'query string')
        elif method == 'POST':
            if headers['content-type'] == 'application/x-www-form-urlencoded':
                import sys
                for line in sys.stdin:
                    parser.parse(registry, line, 'form')
            else:
                import journal
                firewall = journal.firewall('opal')
                firewall.log("NYI: unsupported content-type '%s'" % headers['content-type'])
        else:
            import journal
            journal.firewall('opal').log("unknown method '%s'" % method)

        # if we got commandline arguments, parse them as well
        for arg in argv:
            if arg and arg[0] == '?':
                arg = arg[1:]
            parser.parse(registry, arg, 'command line')
            
        return


    def asCGIScript(self, cgi):

        if cgi:
            # get the headers out asap
            self.printHeaders()

            # take care of exception output
            self.initializeTraceback()

            # format journal output
            self.initializeJournal()

        return


    def printHeaders(self):
        print 'Content-type: text/html'
        print ''

        # just in case further output is done by a subprocess
        import sys
        sys.stdout.flush()

        return


    def initializeTraceback(self):
        # pipe stderr to stdout
        import sys
        sys.stderr = sys.stdout
        
        # decorate exceptions
        import cgitb
        cgitb.enable()
        return


    def initializeJournal(self):
        import journal
        renderer = journal.journal().device.renderer
        renderer.header = '<pre>' + renderer.header
        renderer.footer = renderer.footer + '</pre>'
        return


    def __init__(self, name, asCGI=None):
        Script.__init__(self, name)
        self.stream = None

        if asCGI is None:
            asCGI = True
        self.asCGIScript(asCGI)
        
        return


    def _configure(self):
        Script._configure(self)

        self.stream = self.inventory.stream
        return


    def _createCGIParser(self):
        import opal.applications
        return opal.applications.cgiParser()
        

# version
__id__ = "$Id: CGI.py,v 1.1.1.1 2005/03/15 06:09:10 aivazis Exp $"

# End of file 
