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


from pyre.applications import SuperScript


def mpirun():
    # only import this as needed
    from mpi.scripts import mpirun
    return mpirun()


class PyreScript(SuperScript):


    name = "pyre"


    subscripts = {
        "mpirun": mpirun,
        }


if __name__ == '__main__':
    from pyre.scripts import pyre
    pyre()


# end of file
