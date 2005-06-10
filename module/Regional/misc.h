// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//=====================================================================
//
//                             CitcomS.py
//                 ---------------------------------
//
//                              Authors:
//            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
//          (c) California Institute of Technology 2002-2005
//
//        By downloading and/or installing this software you have
//       agreed to the CitcomS.py-LICENSE bundled with this software.
//             Free for non-commercial academic research ONLY.
//      This program is distributed WITHOUT ANY WARRANTY whatsoever.
//
//=====================================================================
//
//  Copyright June 2005, by the California Institute of Technology.
//  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
// 
//  Any commercial use must be negotiated with the Office of Technology
//  Transfer at the California Institute of Technology. This software
//  may be subject to U.S. export control laws and regulations. By
//  accepting this software, the user agrees to comply with all
//  applicable U.S. export laws and regulations, including the
//  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
//  the Export Administration Regulations, 15 C.F.R. 730-744. User has
//  the responsibility to obtain export licenses, or other export
//  authority as may be required before exporting such information to
//  foreign countries or providing access to foreign nationals.  In no
//  event shall the California Institute of Technology be liable to any
//  party for direct, indirect, special, incidental or consequential
//  damages, including lost profits, arising out of the use of this
//  software and its documentation, even if the California Institute of
//  Technology has been advised of the possibility of such damage.
// 
//  The California Institute of Technology specifically disclaims any
//  warranties, including the implied warranties or merchantability and
//  fitness for a particular purpose. The software and documentation
//  provided hereunder is on an "as is" basis, and the California
//  Institute of Technology has no obligations to provide maintenance,
//  support, updates, enhancements or modifications.
//
//=====================================================================
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_misc_h)
#define pyCitcom_misc_h

// copyright
extern char pyCitcom_copyright__name__[];
extern char pyCitcom_copyright__doc__[];
extern "C"
PyObject * pyCitcom_copyright(PyObject *, PyObject *);


extern char pyCitcom_return1_test__name__[];
extern char pyCitcom_return1_test__doc__[];
extern "C"
PyObject * pyCitcom_return1_test(PyObject *, PyObject *);


extern char pyCitcom_read_instructions__name__[];
extern char pyCitcom_read_instructions__doc__[];
extern "C"
PyObject * pyCitcom_read_instructions(PyObject *, PyObject *);


extern char pyCitcom_CPU_time__name__[];
extern char pyCitcom_CPU_time__doc__[];
extern "C"
PyObject * pyCitcom_CPU_time(PyObject *, PyObject *);


//
//

extern char pyCitcom_citcom_init__doc__[];
extern char pyCitcom_citcom_init__name__[];
extern "C"
PyObject * pyCitcom_citcom_init(PyObject *, PyObject *);


extern char pyCitcom_global_default_values__name__[];
extern char pyCitcom_global_default_values__doc__[];
extern "C"
PyObject * pyCitcom_global_default_values(PyObject *, PyObject *);


extern char pyCitcom_set_signal__name__[];
extern char pyCitcom_set_signal__doc__[];
extern "C"
PyObject * pyCitcom_set_signal(PyObject *, PyObject *);


extern char pyCitcom_velocities_conform_bcs__name__[];
extern char pyCitcom_velocities_conform_bcs__doc__[];
extern "C"
PyObject * pyCitcom_velocities_conform_bcs(PyObject *, PyObject *);


extern char pyCitcom_BC_update_plate_velocity__name__[];
extern char pyCitcom_BC_update_plate_velocity__doc__[];
extern "C"
PyObject * pyCitcom_BC_update_plate_velocity(PyObject *, PyObject *);


extern char pyCitcom_Tracer_tracer_advection__name__[];
extern char pyCitcom_Tracer_tracer_advection__doc__[];
extern "C"
PyObject * pyCitcom_Tracer_tracer_advection(PyObject *, PyObject *);


extern char pyCitcom_Visc_update_material__name__[];
extern char pyCitcom_Visc_update_material__doc__[];
extern "C"
PyObject * pyCitcom_Visc_update_material(PyObject *, PyObject *);


#endif

// version
// $Id: misc.h,v 1.18 2005/06/10 02:23:19 leif Exp $

// End of file
