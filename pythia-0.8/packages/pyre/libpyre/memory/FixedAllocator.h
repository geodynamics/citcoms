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

#if !defined(pyre_memory_FixedAllocator_h)
#define pyre_memory_FixedAllocator_h

// place in the memory namespace
namespace pyre {
    namespace memory {
        template <size_t slots, size_t slotSize> class FixedAllocator;
    }
}

template <size_t slots, size_t slotSize>
class pyre::memory::FixedAllocator {
// interface
public:
    inline void * allocate();

    inline void * end();
    inline void * begin();

    inline void * next() const;
    inline const void * end() const;
    inline const void * begin() const;

    inline bool valid(void * slot) const;

// meta-methods
public:
    inline ~FixedAllocator();
    inline FixedAllocator();

private:
    FixedAllocator(const FixedAllocator &);
    const FixedAllocator & operator=(const FixedAllocator &);

private:
    // store the bin capacity
    static const size_t _capacity = slots * slotSize;

    // the actual memory buffer:
    //     char * for easy arithmetic
    //     let the clients decide where the bin is placed (stack or heap)
    // no alignment problems detected. however, if this starts core dumping...
    char _bin[_capacity + slotSize]; 

    // the cursor to the next available slot
    char * _next;
};

// include the inlines
#include "FixedAllocator.icc"

#endif

// version
// $Id: FixedAllocator.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
