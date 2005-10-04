// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
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
// $Id$

// End of file
