// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Source_h)
#define pyCitcom_Source_h

#include <memory>
#include <vector>
#include "Array2D.h"
#include "Interpolator.h"
#include "mpi.h"


struct All_variables;
class BoundedMesh;


class Source {
protected:
    MPI_Comm comm;
    const int sink;
    std::auto_ptr<Interpolator> interp;
    Array2D<int,1> meshNode_;

public:
    Source(MPI_Comm comm, int sink,
	   BoundedMesh& mesh, const All_variables* E,
	   const BoundedBox& mybbox);
    virtual ~Source() {};

    inline int size() const {return meshNode_.size();}
    void interpolateT(Array2D<double,1>& T, const All_variables* E) const;
    void interpolateV(Array2D<double,DIM>& V, const All_variables* E) const;

    template <class T, int N>
    void sendArray2D(const Array2D<T,N>& array) const;

    template <class T1, int N1, class T2, int N2>
    void sendArray2D(const Array2D<T1,N1>& array1,
		     const Array2D<T2,N2>& array2) const;

private:
    void recvMesh(BoundedMesh& mesh);
    void sendMeshNode() const;

};


template <class T, int N>
void Source::sendArray2D(const Array2D<T,N>& array) const
{
    if(size()) {
	MPI_Request request;
	// are you sure size() == array.size()?
	array.send(comm, sink, 0, array.size(), request);
	array.wait(request);
    }
}


template <class T1, int N1, class T2, int N2>
void Source::sendArray2D(const Array2D<T1,N1>& array1,
			 const Array2D<T2,N2>& array2) const
{
    if(size()) {
	std::vector<MPI_Request> request;
	request.reserve(2);

	request.push_back(MPI_Request());
	array1.send(comm, sink, 0, array1.size(), request.back());

	request.push_back(MPI_Request());
	array2.send(comm, sink, 0, array2.size(), request.back());

	array1.wait(request);
    }
}


#endif

// version
// $Id: Source.h,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
