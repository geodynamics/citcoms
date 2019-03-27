# TODO: change MODULE to the correct name of the new module.
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this module'''

    text = '''#=====================================================================
# example config.cfg file for the MODULE
# ... 
# 
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
def test( argv ):
    '''geodynamic framework module self test'''
    global verbose
    verbose = True 
    print(now(), 'test: sys.argv = ', sys.argv )
    # run the tests 

    # read the defaults
    frame_d = parse_geodynamic_framework_defaults()

    # read the first command line argument as a .cfg file 
    cfg_d = parse_configuration_file( sys.argv[1] )

#=====================================================================
#=====================================================================
if __name__ == "__main__":
    import MODULE

    if len( sys.argv ) > 1:

        # make the example configuration file 
        if sys.argv[1] == '-e':
            make_example_config_file()
            sys.exit(0)

        # process sys.arv as file names for testing 
        test( sys.argv )
    else:
        # print module documentation and exit
        # TODO : replace with actual module name :
        # help(Core_Util)

#=====================================================================
#=====================================================================
