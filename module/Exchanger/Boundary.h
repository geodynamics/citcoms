// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Boundary_h)
#define pyCitcom_Boundary_h

#include "BoundedMesh.h"

struct All_variables;


class Boundary : public BoundedMesh {
    Array2D<int,DIM> normal_;

public:
    Boundary();
    explicit Boundary(const All_variables* E);
    virtual ~Boundary() {};

    inline int normal(int d, int n) const {return normal_[d][n];}

    virtual void broadcast(const MPI_Comm& comm, int broadcaster);
    virtual void broadcast(const MPI_Comm& comm, int broadcaster) const;
    virtual void recv(const MPI_Comm& comm, int sender);
    virtual void send(const MPI_Comm& comm, int receiver) const;

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E);

};


#endif

// version
// $Id: Boundary.h,v 1.29 2004/03/28 23:01:57 tan2 Exp $

// End of file
