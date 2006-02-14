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

#if !defined(pyre_geometry_Point_h)
#define pyre_geometry_Point_h

#include "../containers/Tuple.h"

// place in the geometry namespace
namespace pyre {
    namespace geometry {
        template <size_t dim, typename scalar_t> class Point;
    }
}

template <size_t dim = 3, typename Scalar_t = double>
class pyre::geometry::Point : public pyre::containers::Tuple<Scalar_t, dim>
{
// types
public:
    typedef Scalar_t scalar_t;
    typedef typename pyre::containers::Tuple<scalar_t, dim>::initializer_t initializer_t;
    static const size_t dimension = dim;

// interface
public:
    initializer_t operator= (scalar_t item);

// meta-methods
public:
    inline Point();
    inline ~Point();

// disable these
private:
    Point(const Point<dim, scalar_t> &);
    const Point<dim, scalar_t> & operator=(const Point<dim, scalar_t> &);
};

// get the definitions
#include "Point.icc"

#endif

// version
// $Id: Point.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
