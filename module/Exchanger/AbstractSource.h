// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_AbstractSource_h)
#define pyCitcom_AbstractSource_h

#include <vector>
#include "mpi.h"
#include "utility.h"
#include "Array2D.h"
#include "FEMInterpolator.h"


class AbstractSource {
protected:
    MPI_Comm comm;
    const int sink;
    Array2D<int,1> meshNode_;
    FEMInterpolator* interp;

public:
    AbstractSource(MPI_Comm c, int s) : comm(c), sink(s), interp(NULL) {};
    virtual ~AbstractSource() {delete interp;}

    inline int size() const {return meshNode_.size();}

    virtual void interpolateForce(Array2D<double,DIM>& F) const = 0;
    virtual void interpolatePressure(Array2D<double,1>& P) const = 0;
    virtual void interpolateStress(Array2D<double,DIM>& S) const = 0;
    virtual void interpolateTemperature(Array2D<double,1>& T) const = 0;
    virtual void interpolateTraction(Array2D<double,DIM>& F) const = 0;
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const = 0;

    template <class T, int N>
    void send(const Array2D<T,N>& array) const;

    template <class T1, int N1, class T2, int N2>
    void send(const Array2D<T1,N1>& array1,
	      const Array2D<T2,N2>& array2) const;

private:

    // disable copy c'tor and assignment operator
    AbstractSource(const AbstractSource&);
    AbstractSource& operator=(const AbstractSource&);

};


template <class T, int N>
void AbstractSource::send(const Array2D<T,N>& array) const
{
#ifdef DEBUG
    if(size() != array.size()) {
	journal::firewall_t firewall("AbstractSource");
	firewall << journal::loc(__HERE__)
		 << "AbstractSource: inconsistenet array size" << journal::end;
	throw std::out_of_range("AbstractSource");
    }
#endif

    if(size()) {
	MPI_Request request;
	array.send(comm, sink, 0, array.size(), request);
	util::waitRequest(request);
    }
}


template <class T1, int N1, class T2, int N2>
void AbstractSource::send(const Array2D<T1,N1>& array1,
			  const Array2D<T2,N2>& array2) const
{
#ifdef DEBUG
    if(size() != array1.size() || size() != array2.size()) {
	journal::firewall_t firewall("AbstractSource");
	firewall << journal::loc(__HERE__)
		 << "AbstractSource: inconsistenet array size" << journal::end;
	throw std::out_of_range("AbstractSource");
    }
#endif

    if(size()) {
	std::vector<MPI_Request> request;
	request.reserve(2);

	request.push_back(MPI_Request());
	array1.send(comm, sink, 0, array1.size(), request.back());

	request.push_back(MPI_Request());
	array2.send(comm, sink, 0, array2.size(), request.back());

	util::waitRequest(request);
    }
}


#endif

// version
// $Id: AbstractSource.h,v 1.1 2003/11/25 02:59:11 tan2 Exp $

// End of file
