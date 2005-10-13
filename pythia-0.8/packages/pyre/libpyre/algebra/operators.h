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

#if !defined(pyre_algebra_operators_h)
#define pyre_algebra_operators_h


namespace pyre {
    namespace algebra {
        template <typename numeric_t> numeric_t operator+(const numeric_t &, const numeric_t &);
        template <typename numeric_t> numeric_t operator-(const numeric_t &, const numeric_t &);
    }
}

// include the inlines
#include "operators.icc"
#endif


// version
// $Id: operators.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
