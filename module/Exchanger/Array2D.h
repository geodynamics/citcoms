// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Array2D_h)
#define pyCitcom_Array2D_h

#include <string>
#include <vector>
#include "mpi.h"


template <class T, int N>
class Array2D {
    std::vector<T> a_;

public:
    Array2D();
    explicit Array2D(const int size);
    Array2D(const Array2D<T,N>& rhs);
    ~Array2D();

    inline Array2D<T,N>& operator=(const Array2D<T,N>& rhs);
    inline void swap(Array2D<T,N>& rhs);
    inline void reserve(const int n);
    inline void resize(const int n);
    inline void shrink();
    inline int size() const;

    void send(const MPI_Comm comm, const int receiver) const;
    void receive(const MPI_Comm comm, const int sender);
    void broadcast(const MPI_Comm comm, const int broadcaster);
    void print(const std::string& prefix="Array2D") const;

    class Array1D;  // forward declaration

    inline Array1D operator[](const int index) {return Array1D(a_, index);}
    inline const Array1D operator[](const int index) const {
	return Array1D(const_cast<std::vector<T>&>(a_), index);
    }

    // proxy class for operator[]
    class Array1D {
	std::vector<T>& p_;
	int n_;
    public:
	inline Array1D(std::vector<T>& a_, const int n) : p_(a_), n_(n) {};

	inline T& operator[](const int index) {return p_[index*N+n_];}
	inline const T& operator[](const int index) const {return p_[index*N+n_];}
    };

private:
    static MPI_Datatype typeofT();

};



template <class T, int N>
void swap(Array2D<T,N>& lhs, Array2D<T,N>& rhs);


#include "Array2D.cc"

#endif

// version
// $Id: Array2D.h,v 1.8 2003/10/28 01:51:05 tan2 Exp $

// End of file
