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
    from CitcomS.Components.Const import Const
    from CitcomS.Components.Visc import Visc

    import mpi
    Regional.citcom_init(mpi.world().handle())

    const=Const()
    print dir(const.inventory)
    #print const.inventory.radius.value
    Regional.Const_set_properties(const.inventory)

    #visc=Visc()
    #print dir(visc.inventory) 
    #print visc.inventory.sdepv_expt
    #Regional.Visc_set_properties(visc.inventory)

    Regional.set_signal()
    Regional.set_convection_defaults()

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

    print
    print "Time is %f" % Regional.CPU_time()

    import os, sys
    filename = sys.argv[1]
    print filename
    print "read_instructions:", Regional.read_instructions(filename)

    

# version
__id__ = "$Id: signon.py,v 1.10 2003/07/24 17:46:47 tan2 Exp $"

#  End of file 
