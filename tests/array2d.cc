// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <string>
#include "journal/journal.h"
#include "../module/Exchanger/Array2D.h"

using namespace std;

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
	info << "c'tor" << journal::end;
	d1.print(name);
	i1.print(name);

	info << "copy c'tor" << journal::end;
	d1[0][0] = 1;
	Array2D<double,3> d2(d1);
	d2.print(name);

	info << "op'tor [][]" << journal::end;
	for (int i=0; i<dim; i++)
	    d2[i][1] = 10*i;

	info << d2[1][1] << journal::end;
	d2.print(name);

	info << "swap" << journal::end;
	d2.swap(d1);
	d1.print(name);
	d2.print(name);

	info << "resize" << journal::end;
	i1.resize(3);
	i1.print(name);

	info << "push_back" << journal::end;
	i1.push_back(3);
	i1.push_back(4);
	i1.print(name);
	d1.push_back(vector<double>(dim, 5));
	d1.print(name);

	info << "reserve shrink capacity" << journal::end;
	i1.reserve(1000);
	info << "capacity = " << i1.capacity() << journal::end;
	i1.shrink();
	info << "capacity = " << i1.capacity() << journal::end;

    }

    // testing send/receive ...

    if (rank == 0) {
	d1.sendSize(comm, 1);
    } else if(rank == 1) {
	info << "sendSize/receiveSize" << journal::end;
	Array2D<double,dim> d3;
	info << "received size = " << d3.receiveSize(comm, 0) << journal::end;
    }

    if (rank == 0) {
	d1.send(comm, 1);
    } else if(rank == 1) {
	info << "blocking send -> blocking receive" << journal::end;
	Array2D<double,dim> d3;
	d3.receive(comm, 0);
	d3.print(name);
    }

    if (rank == 0) {
	MPI_Request request;
	MPI_Status status;
	d1.send(comm, 1, request);
	MPI_Wait(&request, &status);
    } else if(rank == 1) {
	info << "non-blocking send -> non-blocking receive" << journal::end;
	Array2D<double,dim> d3;
	MPI_Request request;
	MPI_Status status;
	d3.receive(comm, 0, request);
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
	info << "non-blocking partial send -> non-blocking partial receive" << journal::end;
	Array2D<double,dim> d3(dim*dim);
	vector<MPI_Request> request(dim);
	vector<MPI_Status> status(dim);

	for(int i=0; i<dim; i++)
	    d3.receive(comm, 0, i*dim, dim, request[i]);

	MPI_Waitall(dim, &request[0], &status[0]);
	d3.print(name);
    }

    MPI_Finalize();
}

// version
// $Id: array2d.cc,v 1.1 2003/10/30 22:25:03 tan2 Exp $

// End of file
