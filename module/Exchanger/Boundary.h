// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_Boundary_h)
#define pyCitcomSExchanger_Boundary_h

#include "Exchanger/Boundary.h"

struct All_variables;


class Boundary : public Exchanger::Boundary {

public:
    Boundary();
    explicit Boundary(const All_variables* E,
		      bool excludeTop=false,
		      bool excludeBottom=false);
    virtual ~Boundary();

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E,
	       bool excludeTop, bool excludeBottom);

    bool checkSidewalls(const All_variables* E,
			int j, int k, std::vector<int>& normalFlag);
    int ijk2node(const All_variables* E,
		 int i, int j, int k);
    void appendNode(const All_variables* E,
		    int node, const std::vector<int>& normalFlag);
};


#endif

// version
// $Id: Boundary.h,v 1.31 2004/05/28 21:26:34 tan2 Exp $

// End of file
