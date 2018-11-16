#!/usr/bin/env python

import os.path

def parsemodel(somepath):
    
    nothing = (None, None, None, None)

    # prepare...
    filepath = os.path.abspath(somepath)

    # first round
    pardir, filename = os.path.split(filepath)
    if not filename:
        return nothing

    # second round
    rootname, h5 = os.path.splitext(filename)
    if not rootname:
        return nothing
    if h5 != '.h5':
        return nothing

    # third round
    modelname, dotstep = os.path.splitext(rootname)
    if not modelname:
        return nothing

    coordfile = '%s.h5' % modelname
    coordpath = os.path.join(pardir, coordfile)

    # fourth round
    try:
        step = int(dotstep[1:])
    except ValueError:
        return (None, modelname, coordpath, None)

    return (step, modelname, coordpath, filepath)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print parsemodel(sys.argv[1])

