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


from pyre.launchers.Launcher import Launcher as Base


class Launcher(Base):
    
    
    import pyre.inventory as pyre

    dry = pyre.bool("dry", default=False)
    dry.meta['tip'] = "prints the command line and exits"
        
    nodegen = pyre.str("nodegen")
    nodegen.meta['tip'] = """a printf-style format string, used in conjunction with 'nodelist' to generate the list of machine names (e.g., "n%03d")"""
        
    command = pyre.str("command", default="mpirun -np ${nodes}")


    def launch(self):
        import os, sys

        argv = self.argv()
        command = ' '.join(argv)
        
        if self.dry:
            print command
            return
        
        self._info.log("spawning: %s" % command)

        # The following is based upon os.spawnvp() internals.
        status = None
        pid = os.fork()
        if not pid:
            # Child
            try:
                os.execvp(argv[0], argv)
            except Exception, e:
                # See Issue116.
                print >>sys.stderr, 'execvp("%s"): %s' % (argv[0], e)
                os._exit(127)
        else:
            # Parent
            while 1:
                wpid, sts = os.waitpid(pid, 0)
                if os.WIFSTOPPED(sts):
                    continue
                elif os.WIFSIGNALED(sts):
                    status = -os.WTERMSIG(sts)
                    break
                elif os.WIFEXITED(sts):
                    status = os.WEXITSTATUS(sts)
                    break
                else:
                    assert False, "Not stopped, signaled or exited???"
        
        statusMsg = "%s: %s: exit %d" % (sys.executable, argv[0], status)
        if status != 0:
            sys.exit(statusMsg)
        self._info.log(statusMsg)

        return


    def argv(self): return self._buildArgumentList()


    def _buildArgumentList(self):
        import os
        
        if not self.nodes:
            self.nodes = len(self.nodelist)

        if self.nodes < 1:
            self.nodes = 1

        # Environment variable references such as ${PBS_NODEFILE} are
        # allowed in 'command', thanks to the Preprocessor component.
        # See mpi.Application.getStateArgs() to see how the ${nodes}
        # macro is defined.
        args = self.command.split()
        
        # use only the specific nodes specified explicitly
        if self.nodelist:
            self._expandNodeListArgs(args)

        args.append(os.path.abspath(self.executable))
        args += self.arguments

        return args


# end of file 
