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
    double fge_t;
    double cge_t;

public:
    Inlet(const BoundedMesh& boundedMesh, const Sink& s, All_variables* e);
    virtual ~Inlet();

    void storeTimestep(double fge_t, double cge_t);
    virtual void recv() = 0;
    virtual void impose() = 0;

protected:
    void getFactor(double& N1, double& N2) const;

private:
    // disable copy c'tor and assignment operator
    Inlet(const Inlet&);
    Inlet& operator=(const Inlet&);

};


#endif

// version
// $Id: Inlet.h,v 1.2 2004/03/11 01:06:14 tan2 Exp $

// End of file
