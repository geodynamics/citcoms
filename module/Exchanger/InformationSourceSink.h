// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_InformationSourceSink_h)
#define pyCitcom_InformatioSourceSinkn_h

#include "mpi.h"
#include "BoundedMesh.h"
#include "Sink.h"
#include "Source.h"


class InformationSink {
protected:
    const BoundedMesh* mesh;
    const Sink* sink;

public:
    InformationSink(const BoundedMesh* m) : mesh(m), sink(NULL) {};
    virtual ~InformationSink() {delete mesh; delete sink;}

    virtual void recv() = 0;
    virtual void impose() = 0;

private:
    // disable copy c'tor and assignment operator
    InformationSink(const InformationSink&);
    InformationSink& operator=(const InformationSink&);

};




class InformationSource {
protected:
    BoundedMesh* mesh;
    const AbstractSource* source;

public:
    InformationSource() : mesh(NULL), source(NULL) {};
    virtual ~InformationSource() {};

    virtual void send() = 0;

private:
    // disable copy c'tor and assignment operator
    InformationSource(const InformationSource&);
    InformationSource& operator=(const InformationSource&);

};


#endif

// version
// $Id: InformationSourceSink.h,v 1.1 2003/11/25 02:59:11 tan2 Exp $

// End of file
