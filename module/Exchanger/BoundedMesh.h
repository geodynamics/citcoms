// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_BoundedMesh_h)
#define pyCitcom_BoundedMesh_h

#include <string>
#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"
#include "mpi.h"

struct All_variables;


class BoundedMesh {
protected:
    BoundedBox bbox_;         // domain bounds
    Array2D<double,DIM> X_;   // coordinate
    Array2D<int,1> nodeID_;   // node # in Solver data structure
    bool dimensional_;
    bool transformational_;

public:
    BoundedMesh(bool dimensional, bool transformational);
    virtual ~BoundedMesh() = 0;

    inline int size() const {return X_.size();}

    inline const BoundedBox& bbox() const {return bbox_;}
    inline double X(int d, int n) const {return X_[d][n];}
    inline int nodeID(int n) const {return nodeID_[0][n];}

    virtual void broadcast(const MPI_Comm& comm, int broadcaster);
    virtual void broadcast(const MPI_Comm& comm, int broadcaster) const;

private:
    BoundedMesh(const BoundedMesh&);
    BoundedMesh& operator=(const BoundedMesh&);

};


#endif

// version
// $Id: BoundedMesh.h,v 1.5 2004/01/06 22:40:28 puru Exp $

// End of file
