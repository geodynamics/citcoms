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

#if !defined(pyre_containers_Tuple_h)
#define pyre_containers_Tuple_h

namespace pyre {
    namespace containers {
        template <typename Item_t, size_t items> class Tuple;
    }
}

#include "Initializer.h"

//
template <typename Item_t, size_t items>
class pyre::containers::Tuple
{
// template info
public:
    typedef Item_t item_t;
    typedef Tuple<item_t, items> tuple_t;
    typedef InitializerSwitch<tuple_t, item_t *> initializer_t;
    static const size_t length = items;

// interface
public:
    size_t size() const;

    item_t * data();
    const item_t * data() const;

    void initialize(item_t item);

    item_t & operator()(size_t index);
    item_t operator()(size_t index) const;

    initializer_t operator= (item_t item);

// meta-methods
public:
    ~Tuple();

    Tuple();
    Tuple(const Tuple<item_t, items> &);
    const Tuple<item_t, items> & operator=(const Tuple<item_t, items> & rhs);

private:
    item_t _buffer[length];
};


// include the inlines
#include "Tuple.icc"

#endif

// version
// $Id: Tuple.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
