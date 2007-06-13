#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2007, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''
Paste CitcomS data together

  Usage: pasteCitcomData.py datadir datafile infix1,infix2[,...] rank step save_dir

datadir:   directory of the output files
datafile:  prefix of the output files
infix1, infix2, ...: the infix to be pasted (e.g.: coord, velo)
rank:      MPI rank of the output
step:      time step of the output
save_dir:  directory for the pasted file
'''


def run(datadir, datafile, opts, rank, step, save_dir):

    outfile = '%s/%s.%s.%d.%d.pasted' % (save_dir, datafile, opts, rank, step)
    f = open(outfile, 'w')

    try:
        paste(datadir, datafile, opts, rank, step, f)
    finally:
        f.close()

    return



def paste(datadir, datafile, opts, rank, step, stream=None):
    if stream is None:
        import sys
        stream = sys.stdout

    files = []
    for infix in opts.split(','):
        f = open_file(datadir, datafile, infix, rank, step)
        strip_headerlines(f, infix)
        files.append(f)

    paste_files_and_write(stream, *files)
    return



def expand_datadir(datadir):
    '''Expand the special strings in datadir
    '''

    ##
    s = "%HOSTNAME"
    try:
        datadir.index(s)
    except: pass
    else:
        from socket import gethostname
        datadir = datadir.replace(s, gethostname())

    ##
    s = "%RANK"
    try:
        datadir.index(s)
    except: pass
    else:
        datadir = datadir.replace(s, rank)

    ##
    if datadir == "%DATADIR":
        fp = os.popen("citcoms_datadir", "r")
        datadir = fp.readline().strip()
        fp.close()

    return datadir



def open_file(datadir, datafile, infix, rank, step):

    if infix == 'coord':
        filename = '%s/%s.%s.%d' % (datadir, datafile, infix, rank)
    else:
        filename = '%s/%s.%s.%d.%d' % (datadir, datafile, infix, rank, step)

    f = open(filename, 'r')
    return f



def strip_headerlines(f, infix):
    '''Remove the header lines from f
    '''

    # how many header lines for each infix
    headers = {'coord': 1,
               'botm': 1,
               'comp_nd': 1,
               'pressure': 2,
               'stress': 2,
               'surf': 1,
               'velo': 2,
               'visc': 1}

    nlines = headers[infix]
    for i in range(nlines):
        f.readline()

    return



def paste_files_and_write(stream, *files):
    # zip all file iterators in one
    from itertools import izip
    lines = izip(*files)

    # read all files simulataneously
    for line in lines:
        line = ' '.join([x.strip() for x in line]) + '\n'
        stream.write(line)

    return



if __name__ == '__main__':

    import sys

    if len(sys.argv) < 6:
        print __doc__
        sys.exit()


    datadir = expand_datadir(sys.argv[1])
    datafile = sys.argv[2]
    opts = sys.argv[3]
    rank = int(sys.argv[4])
    step = int(sys.argv[5])
    save_dir = sys.argv[6]

    run(datadir, datafile, opts, rank, step, save_dir)


# End of file
