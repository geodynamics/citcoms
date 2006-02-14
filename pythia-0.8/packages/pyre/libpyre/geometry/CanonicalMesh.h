// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyre_geometry_CanonicalMesh_h)
#define pyre_geometry_CanonicalMesh_h

// place in the geometry namespace
namespace pyre {
    namespace geometry {
        template <typename scalar_t> class CanonicalMesh;
    }
}

template <typename Scalar_t = double>
class pyre::geometry::CanonicalMesh
{
// types
public:
    typedef Scalar_t scalar_t;

// interface
public:

    inline size_t dim() const;
    inline size_t order() const;

    inline size_t vertexCount() const;
    inline size_t simplexCount() const;

    inline void vertexCount(size_t);
    inline void simplexCount(size_t);

    inline void vertex(long id, size_t axis, scalar_t coordinate);
    inline void simplex(long id, size_t node, long vertex);

    inline scalar_t vertex(long id, size_t axis) const;
    inline long simplex(long id, size_t vertex) const;

    inline scalar_t * vertices();
    inline long * simplices();

// meta-methods
public:
    inline CanonicalMesh(size_t dim=3, size_t simplexOrder=4);
    inline ~CanonicalMesh();

// disable these
private:
    CanonicalMesh(const CanonicalMesh<scalar_t> &);
    const CanonicalMesh<scalar_t> & operator=(const CanonicalMesh<scalar_t> &);

// data
private:
    size_t _dim;
    size_t _order;
    size_t _nVertices;
    size_t _nSimplices;
    scalar_t * _vertices;
    long * _simplices;
};

// get the definitions
#include "CanonicalMesh.icc"

#endif

// version
// $Id: CanonicalMesh.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
