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

#if !defined(pyre_picklers_tecplot_mesh_h)
#define pyre_picklers_tecplot_mesh_h

#include <string>
#include <sstream>

#include "pyre/geometrical/CanonicalMesh.h"

// declare in the right namespace

namespace pyre {
    namespace picklers {
        namespace tecplot {
            template <typename scalar_t>
            std::string mesh(const pyre::geometry::CanonicalMesh<scalar_t> & mesh);
        }
    }
}


// the definition

template <typename scalar_t>
std::string pyre::picklers::tecplot::mesh(const pyre::geometry::CanonicalMesh<scalar_t> & mesh) {
    std::ostringstream out;

    return out.str();
}

#endif

// version
// $Id: mesh.h,v 1.1.1.1 2005/03/08 16:13:51 aivazis Exp $

// End of file
