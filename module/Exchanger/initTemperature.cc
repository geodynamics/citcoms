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
#include "global_defs.h"
#include "journal/journal.h"
#include "Exchanger/BoundedBox.h"
#include "initTemperature.h"

using Exchanger::BoundedBox;

void hot_blob(const BoundedBox& bbox, All_variables* E);
void hot_blob_below(const BoundedBox& bbox, All_variables* E);
void five_hot_blobs(const BoundedBox& bbox, All_variables* E);
void add_hot_blob(All_variables* E,
		  double x_center, double y_center, double z_center,
		  double radius, double baseline, double amp);
void debug_output(const All_variables* E);
void basal_tbl_central_hot_blob(const BoundedBox& bbox, All_variables* E);


extern "C" {
    void temperatures_conform_bcs(struct All_variables*);
}


void initTemperature(const BoundedBox& bbox, All_variables* E)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    hot_blob_below(bbox, E);
    //hot_blob(bbox, E);
    //five_hot_blobs(bbox, E);

    debug_output(E);
}


void hot_blob(const BoundedBox& bbox, All_variables* E)
{
    // put a hot blob in the center of fine grid mesh and T=0 elsewhere

    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int i=1; i<E->lmesh.nno; ++i)
	    E->T[m][i] = 0;

    const double theta_min = bbox[0][0];
    const double theta_max = bbox[1][0];
    const double fi_min = bbox[0][1];
    const double fi_max = bbox[1][1];
    const double ri = bbox[0][2];
    const double ro = bbox[1][2];

    // radius of the blob is one third of the smallest dimension
    double d = std::min(std::min(theta_max - theta_min,
				 fi_max - fi_min),
                        ro - ri) / 3;

    // center of fine grid mesh
    double theta_center = 0.5 * (theta_max + theta_min);
    double fi_center = 0.5 * (fi_max + fi_min);
    double r_center = 0.5 * (ro + ri);

    double x_center = r_center * sin(fi_center) * cos(theta_center);
    double y_center = r_center * sin(fi_center) * sin(theta_center);
    double z_center = r_center * cos(fi_center);

    // compute temperature field according to nodal coordinate
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);
}


void hot_blob_below(const BoundedBox& bbox, All_variables* E)
{
    // put a hot blob below the fine grid mesh and T=0 elsewhere

    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int i=1; i<E->lmesh.nno; ++i)
	    E->T[m][i] = 0;

    const double theta_min = bbox[0][0];
    const double theta_max = bbox[1][0];
    const double fi_min = bbox[0][1];
    const double fi_max = bbox[1][1];

    // radius of the blob
    double d = 0.1;

    // center of fine grid mesh
    double theta_center = 0.5 * (theta_max + theta_min);
    double fi_center = 0.5 * (fi_max + fi_min);
    double r_center = 0.6;

    double x_center = r_center * sin(fi_center) * cos(theta_center);
    double y_center = r_center * sin(fi_center) * sin(theta_center);
    double z_center = r_center * cos(fi_center);

    // compute temperature field according to nodal coordinate
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);
}


void five_hot_blobs(const BoundedBox& bbox, All_variables* E)
{
    // put a hot blob in the center of fine grid mesh and T=0 elsewhere
    // also put 4 hot blobs around bbox in coarse mesh

    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int i=1; i<E->lmesh.nno; ++i)
	    E->T[m][i] = 0;

    const double theta_min = bbox[0][0];
    const double theta_max = bbox[1][0];
    const double fi_min = bbox[0][1];
    const double fi_max = bbox[1][1];
    const double ri = bbox[0][2];
    const double ro = bbox[1][2];

    // radius of blobs is one third of the smallest dimension
    double d = std::min(std::min(theta_max - theta_min,
				 fi_max - fi_min),
			ro - ri) / 3;

    // center of hot blob is in the center of bbx
    double theta_center = 0.5 * (theta_max + theta_min);
    double fi_center = 0.5 * (fi_max + fi_min);
    double r_center = 0.5 * (ro + ri);
    double x_center = r_center * sin(fi_center) * cos(theta_center);
    double y_center = r_center * sin(fi_center) * sin(theta_center);
    double z_center = r_center * cos(fi_center);
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);

    // center of hot blob is outside bbx
    theta_center = theta_max + 0.4 * (theta_max - theta_min);
    x_center = r_center * sin(fi_center) * cos(theta_center);
    y_center = r_center * sin(fi_center) * sin(theta_center);
    z_center = r_center * cos(fi_center);
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);

    theta_center = theta_min - 0.4 * (theta_max - theta_min);
    x_center = r_center * sin(fi_center) * cos(theta_center);
    y_center = r_center * sin(fi_center) * sin(theta_center);
    z_center = r_center * cos(fi_center);
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);

    theta_center = 0.5 * (theta_max + theta_min);
    fi_center = fi_max + 0.4 * (fi_max - fi_min);
    x_center = r_center * sin(fi_center) * cos(theta_center);
    y_center = r_center * sin(fi_center) * sin(theta_center);
    z_center = r_center * cos(fi_center);
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);

    fi_center = fi_min - 0.4 * (fi_max - fi_min);
    x_center = r_center * sin(fi_center) * cos(theta_center);
    y_center = r_center * sin(fi_center) * sin(theta_center);
    z_center = r_center * cos(fi_center);
    add_hot_blob(E, x_center, y_center, z_center, d, 0.5, 0.5);

}


void add_hot_blob(All_variables* E,
		  double x_center, double y_center, double z_center,
		  double radius, double baseline, double amp)
{
    // compute temperature field according to nodal coordinate
    for(int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int k=1;k<=E->lmesh.noy;k++)
            for(int j=1;j<=E->lmesh.nox;j++)
                for(int i=1;i<=E->lmesh.noz;i++)  {
                    int node = i + (j-1)*E->lmesh.noz
                             + (k-1)*E->lmesh.noz*E->lmesh.nox;

                    double theta = E->sx[m][1][node];
                    double fi = E->sx[m][2][node];
                    double r = E->sx[m][3][node];

		    double x = r * sin(fi) * cos(theta);
                    double y = r * sin(fi) * sin(theta);
                    double z = r * cos(fi);

                    double distance = sqrt((x - x_center)*(x - x_center) +
                                           (y - y_center)*(y - y_center) +
                                           (z - z_center)*(z - z_center));

                    if (distance < radius)
                        E->T[m][node] += baseline
			              + amp * cos(distance/radius * M_PI);

                }
}


void debug_output(const All_variables* E)
{
    journal::debug_t debugInitT("CitcomS-initTemperature");
    debugInitT << journal::loc(__HERE__);

    for(int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int k=1;k<=E->lmesh.noy;k++)
            for(int j=1;j<=E->lmesh.nox;j++)
                for(int i=1;i<=E->lmesh.noz;i++)  {
                    int node = i + (j-1)*E->lmesh.noz
                             + (k-1)*E->lmesh.noz*E->lmesh.nox;

                    double theta = E->sx[m][1][node];
                    double fi = E->sx[m][2][node];
                    double r = E->sx[m][3][node];

		    debugInitT << "(theta,fi,r,T) = "
			       << theta << "  "
			       << fi << "  "
			       << r << "  "
			       << E->T[m][node] << journal::newline;
                }
    debugInitT << journal::end;
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


void modifyT(const BoundedBox& bbox, All_variables* E)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

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
// $Id: initTemperature.cc,v 1.5 2004/05/11 07:55:30 tan2 Exp $

// End of file
