// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include <cmath>
#include <string>
#include "Array2D.h"
#include "Interior.h"
#include "InteriorImposing.h"
#include "Sink.h"
#include "Source.h"
#include "global_defs.h"
#include "initTemperature.h"

void basal_tbl_central_hot_blob(const BoundedBox& bbox, All_variables* E);

extern "C" {
    void temperatures_conform_bcs(struct All_variables*);
}


void initTemperatureSink(const Interior& interior,
			 const Sink& sink,
			 All_variables* E)
{
    InteriorImposingSink itsink(interior, sink, E);
    itsink.recvT();
    itsink.imposeIC();
}


void initTemperatureSource(const Source& source,
			   All_variables* E)
{
    InteriorImposingSource itsource(source, E);
    itsource.sendT();
}


void modifyT(const BoundedBox& bbox, All_variables* E)
{
    basal_tbl_central_hot_blob(bbox, E);
    temperatures_conform_bcs(E);
}


void basal_tbl_central_hot_blob(const BoundedBox& bbox, All_variables* E)
{
    /*
      This function modify/decorate the temperature field.
      It puts a thermal boundary layer (TBL) is the bottom and puts a hot
      anomaly in the center of bbox.
     */

    const double theta_min = bbox[0][0];
    const double theta_max = bbox[1][0];
    const double fi_min = bbox[0][1];
    const double fi_max = bbox[1][1];
    const double ri = bbox[0][2];
    const double ro = bbox[1][2];

    // center of bbox
    const double theta_center = 0.5 * (theta_max + theta_min);
    const double fi_center = 0.5 * (fi_max + fi_min);

    // blending area
    const double blend1 = (theta_max - theta_min) / 8;
    const double blend2 = (fi_max - fi_min) / 8;
    const double blend3 = (ro - ri) / 8;

    // TBL
    const double thickness = (ro -ri) / 10;
    const double wavelength = std::min(theta_max - theta_min,
				       fi_max - fi_min) / 3;
    const double tbot = 1;
    const double tint = 0.5;

    // compute temperature field according to nodal coordinate
    for(int m=1; m<=E->sphere.caps_per_proc; m++)
        for(int k=1; k<=E->lmesh.noy; k++)
            for(int j=1; j<=E->lmesh.nox; j++)
                for(int i=1; i<=E->lmesh.noz; i++)  {
                    int node = i + (j-1)*E->lmesh.noz
                             + (k-1)*E->lmesh.noz*E->lmesh.nox;

                    double theta = E->sx[m][1][node];
                    double fi = E->sx[m][2][node];
                    double r = E->sx[m][3][node];

		    if(r > ro)
			continue;

		    // put a TBL at bottom, blending in orginal field
		    double factor = 1 - std::exp((r - ro)/blend3);
		    E->T[m][node] = E->T[m][node] * (1 - factor)
			+ tint + (tbot - tint)
			* std::exp((ri - r)/thickness)
			* factor;

		    if(!( (theta > theta_max) || (theta < theta_min) ||
			(fi > fi_max) || (fi < fi_min) )) {
			double factor =
			    (1 - std::exp((theta - theta_max)/blend1))
			    * (1 - std::exp((theta_min - theta)/blend1))
			    * (1 - std::exp((fi - fi_max)/blend2))
			    * (1 - std::exp((fi_min - fi)/blend2));

			double thickness2 = thickness
			    * (1 + 0.25 * std::cos((theta - theta_center)
						      / wavelength))
			    * (1 + 0.25 * std::cos((fi - fi_center)
						      / wavelength))
			    * factor;

			double T = E->T[m][node];
			if(thickness2 > 1e-30) {
			    E->T[m][node] = (T - tint) * std::exp((ri - r) *
								  (1/thickness2 -
								   1/thickness))
				          + tint;
			}

		    }
		}
}


// version
// $Id: initTemperature.cc,v 1.3 2004/01/14 00:34:34 tan2 Exp $

// End of file
