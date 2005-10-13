// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyre_containers_Vector_h)
#define pyre_containers_Vector_h

namespace pyre {
    namespace containers {
        template <typename Item_t> class Vector;
    }
}

//
template <typename Item_t>
class pyre::containers::Vector
{
// template info
public:
    typedef Item_t item_t;

// interface
public:
    size_t size() const;

    item_t & operator()(size_t index);
    item_t operator()(size_t index) const;

// meta-methods
public:
    ~Vector();

    Vector(size_t capacity);
    Vector(const Vector<item_t> &);
    const Vector<item_t> & operator=(const Vector<item_t> & rhs);

private:
    size_t _length;
    item_t * _buffer;
};


// include the inlines
#include "Vector.icc"

#endif

// version
// $Id: Vector.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
