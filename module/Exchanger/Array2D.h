// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Array2D_h)
#define pyCitcom_Array2D_h

#include <portinfo>
#include <iostream>
#include <string>
#include "auto_array_ptr.h"
#include "mpi.h"


template <int N>
class Array2D {
    const int size_;
    auto_array_ptr<double> a_;

public:
    explicit Array2D(const int size);
    Array2D(const Array2D<N>& rhs);
    Array2D(auto_array_ptr<double> array, const int size);
    ~Array2D() {};

    Array2D<N>& operator=(const Array2D<N>& rhs);

    inline double operator()(int d, int n) const {return a_[n*N+d];}
    inline int size() const {return size_;}
    inline void swap(Array2D<N>& rhs) {
	a_.swap(rhs.a_);
	std::swap(size_, rhs.size_);
    }

    void send(const MPI_Comm comm, const int receiver) const;
    void receive(const MPI_Comm comm, const int sender);
    void broadcast(const MPI_Comm comm, const int broadcaster);

    void print(const std::string& prefix="Array2D") const;

};



#endif

// version
// $Id: Array2D.h,v 1.2 2003/10/11 00:35:50 tan2 Exp $

// End of file
