# TODO: replace _SCRIPT_ with the name of the new script; remove this line
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# example config.cfg file for _SCRIPT_
# ... 
# 
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
#=====================================================================
if __name__ == "__main__":

    # check for script called wih no arguments
    if len(sys.argv) != 2:
        usage()
        sys.exit(-1)

    # create example config file 
    if sys.argv[1] == '-e':
        make_example_config_file()
        sys.exit(0)

    # run the main script workflow
    main()
    sys.exit(0)
#=====================================================================
#=====================================================================
