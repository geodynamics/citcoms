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
    BoundedBox bbox_;       // domain bounds
    Array2D<double,DIM> X_;   // coordinate
    Array2D<int,1> meshID_;

public:
    BoundedMesh();
    BoundedMesh(const All_variables* E);
    virtual ~BoundedMesh() {};

    inline int dim() const {return DIM;}
    inline int size() const {return X_.size();}

    inline double theta_min() const {return bbox_[0][0];}
    inline double theta_max() const {return bbox_[1][0];}
    inline double fi_min() const {return bbox_[0][1];}
    inline double fi_max() const {return bbox_[1][1];}
    inline double ri() const {return bbox_[0][2];}
    inline double ro() const {return bbox_[1][2];}

    inline const BoundedBox& bbox() const {return bbox_;}
    inline double X(int d, int n) const {return X_[d][n];}
    inline int meshID(int n) const {return meshID_[0][n];}

    virtual void broadcast(const MPI_Comm& comm, int broadcaster);
    virtual void broadcast(const MPI_Comm& comm, int broadcaster) const;

private:
    void initBBox(const All_variables *E);

    BoundedMesh(const BoundedMesh&);
    BoundedMesh& operator=(const BoundedMesh&);

};
#endif

// version
// $Id: BoundedMesh.h,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
