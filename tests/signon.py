#!/usr/bin/env mpipython.exe
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#  <LicenseText>
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

if __name__ == "__main__":

    import CitcomS 
#    import CitcomS.Full as Full
    import CitcomS.Regional as Regional
    from CitcomS.Components.BC import BC
    from CitcomS.Components.Visc import Visc

    import mpi
    Regional.Citcom_Init(mpi.world().handle())

    bc=BC()
    dir(bc.inventory)
    Regional.BC_set_prop(bc.inventory)

    visc=Visc()
    print visc.__dict__
    print visc.inventory.sdepv_expt
    Regional.Visc_set_prop(visc.inventory)

    print "copyright information:"
    print "   ", CitcomS.copyright()

    print
    print "module information:"
#    print "  Full--"
#    print "    file:", Full.__file__
#    print "    doc:", Full.__doc__
#    print "    contents:", dir(Full)
    print
    print "  Regional--"
    print "    file:", Regional.__file__
    print "    doc:", Regional.__doc__
    print "    contents:", dir(Regional)


    print
#    print "Full.return1_test:    ", Full.return1_test()
    print "Regional.return1_test:", Regional.return1_test()

    import mpi
    world = mpi.world()
    print mpi.world
    print "Citcom_Init: return", Regional.Citcom_Init(mpi.mpi.world)

    print
    print "Time is %f" % Regional.CPU_time()

    import os, sys
    filename = sys.argv[1]
    print filename
    print "read_instructions:", Regional.read_instructions(filename)

    

# version
__id__ = "$Id: signon.py,v 1.9 2003/07/13 22:58:12 tan2 Exp $"

#  End of file 
