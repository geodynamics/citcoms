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

#ifdef DEBUG
#include <iostream>
#include <exception>
#include <stdexcept>
#endif

#include <string>
#include <vector>
#include "mpi.h"


template <class T, int N>
class Array2D {
    std::vector<T> a_;

public:
    Array2D();
    explicit Array2D(int size);
    Array2D(const Array2D<T,N>& rhs);
    ~Array2D();

    inline Array2D<T,N>& operator=(const Array2D<T,N>& rhs);
    inline void swap(Array2D<T,N>& rhs);
    inline void reserve(int n);
    inline void resize(int n);
    inline void shrink();
    inline int size() const;
    inline int capacity() const;
    inline bool empty() const;
    inline void push_back(const std::vector<T>& val);
    inline void push_back(const T& val);

    void sendSize(const MPI_Comm& comm, int receiver) const;
    void sendSize(const MPI_Comm& comm, int receiver, int size) const;
    int receiveSize(const MPI_Comm& comm, int sender) const;
    int broadcastSize(const MPI_Comm& comm, int broadcaster) const;

    void send(const MPI_Comm& comm, int receiver) const;
    void send(const MPI_Comm& comm, int receiver, MPI_Request&) const;
    void send(const MPI_Comm& comm, int receiver,
	      int begin, int sendsize, MPI_Request&) const;
    void receive(const MPI_Comm& comm, int sender);
    void receive(const MPI_Comm& comm, int sender, MPI_Request&);
    void receive(const MPI_Comm& comm, int sender,
		 int begin, int recvsize, MPI_Request&);
    void broadcast(const MPI_Comm& comm, int broadcaster);

    void print(const std::string& prefix="Array2D") const;

    class Array1D;  // forward declaration

    inline Array1D operator[](size_t index);
    inline const Array1D operator[](size_t index) const;

    // proxy class for operator[]
    class Array1D {
	std::vector<T>& p_;
	size_t n_;

    public:
	inline Array1D(std::vector<T>& a, size_t n);
	inline T& operator[](size_t index);
	inline const T& operator[](size_t index) const;
    };

private:
    static const int SIZETAG_ = 74;
    static const int TAG_ = 75;
    static MPI_Datatype typeofT();
    static void testResult(int result, const std::string& errmsg);

};


#include "Array2D.cc"

#endif

// version
// $Id: Array2D.h,v 1.12 2003/10/30 22:45:37 tan2 Exp $

// End of file
