// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Inlet_h)
#define pyCitcom_Inlet_h

struct All_variables;
class BoundedMesh;
class Sink;


class Inlet {
protected:
    const BoundedMesh& mesh;
    const Sink& sink;
    All_variables* E;

public:
    Inlet(const BoundedMesh& boundedMesh, const Sink& s, All_variables* e);
    virtual ~Inlet();

    virtual void recv() = 0;
    virtual void impose() = 0;

private:
    // disable copy c'tor and assignment operator
    Inlet(const Inlet&);
    Inlet& operator=(const Inlet&);

};


#endif

// version
// $Id: Inlet.h,v 1.1 2004/02/24 20:03:09 tan2 Exp $

// End of file
