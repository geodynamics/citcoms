//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
//
// <LicenseText>
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <iostream>

#include <Python.h>
#include <mpi.h>


extern "C"
DL_EXPORT(int) Py_Main(int, char **);

int main(int argc, char **argv) {
    int status = MPI_Init(&argc, &argv);
    if (status != MPI_SUCCESS) {
	std::cerr << argv[0] << ": MPI_Init failed! Exiting ..." << std::endl;
	return status;
    }

    status = Py_Main(argc, argv);
    
    MPI_Finalize();
    
    return status;
}

// version
// $Id: mpipython.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
