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
#include "mpi.h"


template <class T, int N>
class Array2D {
    const int size_;
    T* a_;

public:
    explicit Array2D(const int size);
    Array2D(const Array2D<T,N>& rhs);
    Array2D(T* array, const int size);
    ~Array2D();

    Array2D<T,N>& operator=(const Array2D<T,N>& rhs);

    inline int size() const {return size_;}
    void send(const MPI_Comm comm, const int receiver) const;
    void receive(const MPI_Comm comm, const int sender);
    void broadcast(const MPI_Comm comm, const int broadcaster);
    void print(const std::string& prefix="Array2D") const;


    // proxy class for operator[]
    class Array1D {
	T* const& p_;
	const int n_;
    public:
	inline Array1D(T* const& a_, const int n) : p_(a_), n_(n){};

	inline T& operator[](const int index) {
	    return p_[index*N+n_];
	}

	inline const T& operator[](const int index) const {
	    return p_[index*N+n_];
	}
    };

    inline Array1D operator[](const int index) {
	return Array1D(a_, index);
    }

    inline const Array1D operator[](const int index) const {
	return Array1D(a_, index);
    }

private:
    static MPI_Datatype typeofT();
};



#endif

// version
// $Id: Array2D.h,v 1.3 2003/10/16 20:06:02 tan2 Exp $

// End of file
