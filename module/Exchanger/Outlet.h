// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Outlet_h)
#define pyCitcom_Outlet_h


struct All_variables;
class AbstractSource;


class Outlet {
protected:
    const AbstractSource& source;
    All_variables* E;

public:
    Outlet(const AbstractSource& source, All_variables* E);
    virtual ~Outlet();

    virtual void send() = 0;

private:
    // disable copy c'tor and assignment operator
    Outlet(const Outlet&);
    Outlet& operator=(const Outlet&);

};


#endif

// version
// $Id: Outlet.h,v 1.1 2004/02/24 20:03:09 tan2 Exp $

// End of file
