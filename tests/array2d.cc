// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//=====================================================================
//
//                             CitcomS.py
//                 ---------------------------------
//
//                              Authors:
//            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
//          (c) California Institute of Technology 2002-2005
//
//        By downloading and/or installing this software you have
//       agreed to the CitcomS.py-LICENSE bundled with this software.
//             Free for non-commercial academic research ONLY.
//      This program is distributed WITHOUT ANY WARRANTY whatsoever.
//
//=====================================================================
//
//  Copyright June 2005, by the California Institute of Technology.
//  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
// 
//  Any commercial use must be negotiated with the Office of Technology
//  Transfer at the California Institute of Technology. This software
//  may be subject to U.S. export control laws and regulations. By
//  accepting this software, the user agrees to comply with all
//  applicable U.S. export laws and regulations, including the
//  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
//  the Export Administration Regulations, 15 C.F.R. 730-744. User has
//  the responsibility to obtain export licenses, or other export
//  authority as may be required before exporting such information to
//  foreign countries or providing access to foreign nationals.  In no
//  event shall the California Institute of Technology be liable to any
//  party for direct, indirect, special, incidental or consequential
//  damages, including lost profits, arising out of the use of this
//  software and its documentation, even if the California Institute of
//  Technology has been advised of the possibility of such damage.
// 
//  The California Institute of Technology specifically disclaims any
//  warranties, including the implied warranties or merchantability and
//  fitness for a particular purpose. The software and documentation
//  provided hereunder is on an "as is" basis, and the California
//  Institute of Technology has no obligations to provide maintenance,
//  support, updates, enhancements or modifications.
//
//=====================================================================
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <string>
#include <mpi.h>
#include "journal/diagnostics.h"
#include "Array2D.h"

using namespace std;
using namespace Exchanger;

int main(int argc, char** argv)
{
    const string name = "Array2DTest";
    journal::info_t info(name);
    info.activate();

    const int dim = 3;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    Array2D<double,dim> d1(2);
    Array2D<int,1> i1(5);

    if (rank == 0) {
	// testing vector-like behaviors
	info << "c'tor" << journal::endl;
	d1.print(name);
	i1.print(name);

	info << "copy c'tor" << journal::endl;
	d1[0][0] = 1;
	Array2D<double,3> d2(d1);
	d2.print(name);

	info << "op'tor [][]" << journal::endl;
	for (int i=0; i<dim; i++)
	    d2[i][1] = 10*i;

	info << d2[1][1] << journal::endl;
	d2.print(name);

	info << "swap" << journal::endl;
	d2.swap(d1);
	d1.print(name);
	d2.print(name);

	info << "resize" << journal::endl;
	i1.resize(3);
	i1.print(name);

	info << "push_back" << journal::endl;
	i1.push_back(3);
	i1.push_back(4);
	i1.print(name);
	d1.push_back(vector<double>(dim, 5));
	d1.print(name);

	info << "reserve shrink capacity" << journal::endl;
	i1.reserve(1000);
	info << "capacity = " << i1.capacity() << journal::endl;
	i1.shrink();
	info << "capacity = " << i1.capacity() << journal::endl;

    }

    // testing send/receive ...

    if (rank == 0) {
	d1.sendSize(comm, 1);
    } else if(rank == 1) {
	info << "sendSize/receiveSize" << journal::endl;
	Array2D<double,dim> d3;
	info << "received size = " << d3.recvSize(comm, 0) << journal::endl;
    }

    if (rank == 0) {
	d1.send(comm, 1);
    } else if(rank == 1) {
	info << "blocking send -> blocking receive" << journal::endl;
	Array2D<double,dim> d3;
	d3.recv(comm, 0);
	d3.print(name);
    }

    if (rank == 0) {
	MPI_Request request;
	MPI_Status status;
	d1.send(comm, 1, request);
	MPI_Wait(&request, &status);
    } else if(rank == 1) {
	info << "non-blocking send -> non-blocking receive" << journal::endl;
	Array2D<double,dim> d3;
	MPI_Request request;
	MPI_Status status;
	d3.recv(comm, 0, request);
	MPI_Wait(&request, &status);
	d3.print(name);
    }

    if (rank == 0) {
	Array2D<double,dim> d3(dim*dim);
	for(int i=0; i<dim; i++)
	    for(int j=0; j<dim*dim; j++)
		d3[i][j] = (i+1)*j;
	//d3.print(name);

	vector<MPI_Request> request(dim);
	vector<MPI_Status> status(dim);

	for(int i=0; i<dim; i++)
	    d3.send(comm, 1, i*dim, dim, request[i]);

	MPI_Waitall(dim, &request[0], &status[0]);
    } else if(rank == 1) {
	info << "non-blocking partial send -> non-blocking partial receive" << journal::endl;
	Array2D<double,dim> d3(dim*dim);
	vector<MPI_Request> request(dim);
	vector<MPI_Status> status(dim);

	for(int i=0; i<dim; i++)
	    d3.recv(comm, 0, i*dim, dim, request[i]);

	MPI_Waitall(dim, &request[0], &status[0]);
	d3.print(name);
    }

    MPI_Finalize();
}

// version
// $Id: array2d.cc,v 1.3 2005/07/23 01:35:53 leif Exp $

// End of file
