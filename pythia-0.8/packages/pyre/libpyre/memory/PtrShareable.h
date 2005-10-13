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

#if !defined(pyre_memory_PtrShareable_h)
#define pyre_memory_PtrShareable_h

// place in the memory namespace

namespace pyre {
    namespace memory {
        class PtrShareable;
    }
}

template <typename shareable_t>
class pyre::memory::PtrShareable
{
// interface
public:

    inline shareable_t & operator*();
    inline shareable_t * operator->();

// meta-methods
public:
    inline ~PtrShareable();

    inline PtrShareable(shareable_t *);
    inline PtrShareable(const PtrShareable &);
    inline const PtrShareable & operator=(const PtrShareable &);

// implementation
private:
    inline void _init();

private:
	
    shareable_t * _shareable;
};

// include the inlines
#include "PtrShareable.icc"

#endif

// version
// $Id: PtrShareable.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
