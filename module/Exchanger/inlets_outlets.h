// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyExchanger_inlets_outlets_h)
#define pyExchanger_inlets_outlets_h


///////////////////////////////////////////////////////////////////////////////


extern char pyExchanger_BoundaryVTInlet_create__name__[];
extern char pyExchanger_BoundaryVTInlet_create__doc__[];
extern "C"
PyObject * pyExchanger_BoundaryVTInlet_create(PyObject *, PyObject *);


extern char pyExchanger_BoundaryVTInlet_impose__name__[];
extern char pyExchanger_BoundaryVTInlet_impose__doc__[];
extern "C"
PyObject * pyExchanger_BoundaryVTInlet_impose(PyObject *, PyObject *);


extern char pyExchanger_BoundaryVTInlet_recv__name__[];
extern char pyExchanger_BoundaryVTInlet_recv__doc__[];
extern "C"
PyObject * pyExchanger_BoundaryVTInlet_recv(PyObject *, PyObject *);


extern char pyExchanger_BoundaryVTInlet_storeTimestep__name__[];
extern char pyExchanger_BoundaryVTInlet_storeTimestep__doc__[];
extern "C"
PyObject * pyExchanger_BoundaryVTInlet_storeTimestep(PyObject *, PyObject *);


///////////////////////////////////////////////////////////////////////////////


extern char pyExchanger_VTInlet_create__name__[];
extern char pyExchanger_VTInlet_create__doc__[];
extern "C"
PyObject * pyExchanger_VTInlet_create(PyObject *, PyObject *);


extern char pyExchanger_VTInlet_impose__name__[];
extern char pyExchanger_VTInlet_impose__doc__[];
extern "C"
PyObject * pyExchanger_VTInlet_impose(PyObject *, PyObject *);


extern char pyExchanger_VTInlet_recv__name__[];
extern char pyExchanger_VTInlet_recv__doc__[];
extern "C"
PyObject * pyExchanger_VTInlet_recv(PyObject *, PyObject *);


extern char pyExchanger_VTInlet_storeTimestep__name__[];
extern char pyExchanger_VTInlet_storeTimestep__doc__[];
extern "C"
PyObject * pyExchanger_VTInlet_storeTimestep(PyObject *, PyObject *);


///////////////////////////////////////////////////////////////////////////////


extern char pyExchanger_VTOutlet_create__name__[];
extern char pyExchanger_VTOutlet_create__doc__[];
extern "C"
PyObject * pyExchanger_VTOutlet_create(PyObject *, PyObject *);


extern char pyExchanger_VTOutlet_send__name__[];
extern char pyExchanger_VTOutlet_send__doc__[];
extern "C"
PyObject * pyExchanger_VTOutlet_send(PyObject *, PyObject *);


#endif

// version
// $Id: inlets_outlets.h,v 1.1 2004/02/24 20:35:25 tan2 Exp $

// End of file
