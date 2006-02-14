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

#if !defined(pyre_containers_Initializer_h)
#define pyre_containers_Initializer_h

namespace pyre {
    namespace containers {
        template <typename item_t, typename iterator_t> class Initializer;
        template <typename container_t, typename iterator_t> class InitializerSwitch;
    }
}

//
template <typename item_t, typename iterator_t>
class pyre::containers::Initializer
{
// types
    typedef Initializer<item_t, iterator_t> initializer_t;

// interface
public:
    initializer_t operator, (item_t item);

// meta-methods
public:
    ~Initializer();
    Initializer(iterator_t iterator);
    Initializer(const initializer_t &);

// disable these
private:
    Initializer();
    const initializer_t & operator=(const initializer_t & rhs);

private:
    iterator_t _iter;
};

//
template <typename container_t, typename iterator_t>
class pyre::containers::InitializerSwitch
{
// types
public:
    typedef typename container_t::item_t item_t;
    typedef Initializer<item_t, iterator_t> initializer_t;
    typedef InitializerSwitch<container_t, iterator_t> switch_t;

// interface
public:
    void disable() const;
    initializer_t operator, (item_t item);

// meta-methods
public:
    ~InitializerSwitch();
    InitializerSwitch(container_t & container, item_t item);
    InitializerSwitch(const switch_t &);

// disable these
private:
    InitializerSwitch();
    const switch_t & operator= (const switch_t & rhs);

private:
    container_t &  _container;
    item_t _item;
    mutable bool _wipe;
};

// include the inlines
#include "Initializer.icc"

#endif

// version
// $Id: Initializer.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
