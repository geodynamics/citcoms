#!/usr/bin/env python

import subprocess

# generate configure and run
subprocess.call( ['make','clean'] )
subprocess.call( ['libtoolize','--force'] )
subprocess.call( ['aclocal'] )
subprocess.call( ['autoheader'] )
subprocess.call( ['automake', '--force-missing', '--add-missing'] )
subprocess.call( ['autoconf'] )
# I don't know why, but autoreconf -i prevents an error from occuring
subprocess.call( ['autoreconf', '-i'] )
subprocess.call( ['./configure'] )
subprocess.call( ['make'] )
