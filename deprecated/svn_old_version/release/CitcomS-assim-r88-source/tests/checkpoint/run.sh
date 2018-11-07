#!/bin/sh

## Run an ordinary job then restart it. The end result should be the same

../../bin/citcoms tracer.cfg && ../../bin/citcoms tracer.cfg restart.cfg && diff *.tracer.0.5 && diff *.velo.0.5

## clean up
rm yyy.* zzz.* pid*.cfg
