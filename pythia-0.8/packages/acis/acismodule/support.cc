// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "imports"
#include "support.h"

// ACIS includes
#include <faceter/api/af_api.hxx>
#include <faceter/meshmgr/meshmg.hxx>
#include <faceter/meshmgr/idx_mm.hxx>
#include <ga_husk/api/ga_api.hxx>

#include <kernel/kerndata/savres/fileinfo.hxx> 

// helpers
static void setDefaultFileinfo();


bool ACISModeler::initialize()
{
// Initialize the ACIS modeller
    logical success = api_start_modeller(0).ok();

    success &= api_initialize_spline().ok();
    success &= api_initialize_kernel().ok();
    success &= api_initialize_constructors().ok();
    success &= api_initialize_booleans().ok();
    success &= api_initialize_generic_attributes().ok();
    success &= api_initialize_faceter().ok();
    success &= api_initialize_intersectors().ok();

#ifdef ACIS_HAS_MESHER
    success &= api_initialize_mesh_surfaces().ok();
#endif

    // as of ACIS 6.3, a default FileInfo is required
    setDefaultFileinfo();

    // set the save file format to 6.0 so that adlib::acisinterface works
    set_save_file_version(6, 0);

    // done initializing
    return success;
}

void throwACISError(const outcome & check, const char * facility, PyObject * exception)
{
    const char * acis_msg = find_err_mess(check.error_number());
    char * message = new char [strlen(acis_msg) + 128];
    sprintf(message, "%s: error %d: '%s'", facility, check.error_number(), acis_msg);
    
    // set up to throw an exeption
    PyErr_SetString(exception, message);
    
    // clean up
    delete [] message;
    
    return;
}

// helpers
void setDefaultFileinfo()
{
    FileInfo info;

    info.set_product_id("Pyre 1.0");
    info.set_units(1000);

    api_set_file_info(FileId | FileUnits, info);

    return;
}


// version
// $Id: support.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
