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

#if !defined(pyre_memory_Shareable_h)
#define pyre_memory_Shareable_h

// place in the memory namespace

namespace pyre {
    namespace memory {
        class Shareable;
    }
}

class pyre::memory::Shareable
{
// interface
public:
    inline void addReference();
    inline void releaseReference();
    inline int references() const;

// meta-methods
public:
    ~Shareable();

    inline Shareable();
    inline Shareable(const Shareable &);
    inline const Shareable & operator=(const Shareable &);

private:
    int _count;
};

// include the inlines
#include "Shareable.icc"

#endif

// version
// $Id: Shareable.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
