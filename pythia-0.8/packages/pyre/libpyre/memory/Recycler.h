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

#if !defined(pyre_memory_Recycler_h)
#define pyre_memory_Recycler_h

// place in the memory namespace
namespace pyre {
    namespace memory {
        struct Recycler_node_t;
        template <typename node_t> class Recycler;
    }
}


struct pyre::memory::Recycler_node_t { 
    Recycler_node_t * next; 
    Recycler_node_t(Recycler_node_t * next): next(next) {}
};


template <typename Node = pyre::memory::Recycler_node_t>
class pyre::memory::Recycler
{
// types
public:
    typedef Node node_t;

// interface
public:
    inline void * reuse();
    inline void recycle(void * slot);

// meta-methods
public:
    inline ~Recycler();
    inline Recycler();

// disable these
private:
    Recycler(const Recycler &);
    const Recycler & operator=(const Recycler &);

// implementation
private:
    node_t * _recycler;
};

// get the inlines
#include "Recycler.icc"

#endif

// version
// $Id: Recycler.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
