// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//
// data member:
//    bnode_: mapping from CitcomS node # to boundary node #
//    nodeID_: mapping from boundary node # to CitcomS node #
//    X_: coordinate of boundary nodes
//    normal_: normal vector of boundary nodes
//             (for example, if normal = [0, 1, -1], the '0' means this node is
//             not on North-South boundary, the '1' means this node is on East
//             boundary, and the '-1' means this node is on the bottom.



#if !defined(pyCitcomSExchanger_Boundary_h)
#define pyCitcomSExchanger_Boundary_h

#include "Exchanger/Boundary.h"

struct All_variables;


class Boundary : public Exchanger::Boundary {
    std::vector<int> bnode_;

public:
    Boundary();
    explicit Boundary(const All_variables* E,
		      bool excludeTop=false,
		      bool excludeBottom=false);
    virtual ~Boundary();

    inline int bnode(int i) const {return bnode_[i];}

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E,
	       bool excludeTop, bool excludeBottom);

    void addSidewalls(const All_variables* E, int znode, int r_normal);
    bool checkSidewalls(const All_variables* E,
			int j, int k, std::vector<int>& normalFlag);
    int ijk2node(const All_variables* E,
		 int i, int j, int k);
    void appendNode(const All_variables* E,
		    int node, const std::vector<int>& normalFlag);
};


#endif

// version
// $Id: Boundary.h,v 1.33 2004/12/31 01:03:42 tan2 Exp $

// End of file
