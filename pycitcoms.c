/* 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/ 

#include <Python.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "CitcomSmodule.h"

#define COMMAND \
"import sys; " \
"path = sys.argv[1]; " \
"requires = sys.argv[2]; " \
"entry = sys.argv[3]; " \
"path = path.split(':'); " \
"path.extend(sys.path); " \
"sys.path = path; " \
"from pyre.util import loadObject; " \
"entry = loadObject(entry); " \
"entry(sys.argv[3:], kwds={'requires': requires})"

/* include the implementation of _mpi */
#include "mpi/_mpi.c"

struct _inittab inittab[] = {
    { "_mpi", init_mpi },
    { "CitcomSLib", initCitcomSLib },
    { 0, 0 }
};

int main(int argc, char **argv)
{
    int status;
    
#ifdef USE_MPI
    /* initialize MPI */
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "%s: MPI_Init failed! Exiting ...", argv[0]);
        return 1;
    }
#endif
    
    /* add our extension module */
    if (PyImport_ExtendInittab(inittab) == -1) {
        fprintf(stderr, "%s: PyImport_ExtendInittab failed! Exiting...\n", argv[0]);
        return 1;
    }
    
    if (argc < 3 || strcmp(argv[1], "--pyre-start") != 0) {
        return Py_Main(argc, argv);
    }
    
    /* make sure 'sys.executable' is set to the path of this program  */
    Py_SetProgramName(argv[0]);
    
    /* initialize Python */
    Py_Initialize();
    
    /* initialize sys.argv */
    PySys_SetArgv(argc - 1, argv + 1);
    
    /* run the Python command */
    status = PyRun_SimpleString(COMMAND) != 0;
    
    /* shut down Python */
    Py_Finalize();
    
#ifdef USE_MPI
    /* shut down MPI */
    MPI_Finalize();
#endif
    
    return status;
}

/* End of file */
