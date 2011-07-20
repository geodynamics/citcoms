/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 * 
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/* Common functions relating to the building and use of mesh locations ... */

#include <math.h>
#include "global_defs.h"

/* =================================================
  this routine evenly divides the arc between points
  1 and 2 in a great cicle. The word "evenly" means
  anglewise evenly.
 =================================================*/

void even_divide_arc12(int elx, double x1, double y1, double z1, double x2, double y2, double z2, double *theta, double *fi)
{
  double dx,dy,dz,xx,yy,zz;
  int j, nox;

  nox = elx+1;

  dx = (x2 - x1)/elx;
  dy = (y2 - y1)/elx;
  dz = (z2 - z1)/elx;
  for (j=1;j<=nox;j++)   {
      xx = x1 + dx*(j-1) + 5.0e-32;
      yy = y1 + dy*(j-1);
      zz = z1 + dz*(j-1);
      theta[j] = acos(zz/sqrt(xx*xx+yy*yy+zz*zz));
      fi[j]    = myatan(yy,xx);
      }

   return;
  }

/* ================================================
   compute angle and area
   ================================================*/

void compute_angle_surf_area (struct All_variables *E)
{

    int es,el,m,i,j,ii,ia[5],lev;
    double aa,y1[4],y2[4],angle[6],xx[4][5];
    void parallel_process_termination();

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
        ia[1] = 1;
        ia[2] = E->lmesh.noz*E->lmesh.nox-E->lmesh.noz+1;
        ia[3] = E->lmesh.nno-E->lmesh.noz+1;
        ia[4] = ia[3]-E->lmesh.noz*(E->lmesh.nox-1);

        for (i=1;i<=4;i++)  {
            xx[1][i] = E->x[m][1][ia[i]]/E->sx[m][3][ia[1]];
            xx[2][i] = E->x[m][2][ia[i]]/E->sx[m][3][ia[1]];
            xx[3][i] = E->x[m][3][ia[i]]/E->sx[m][3][ia[1]];
        }

        get_angle_sphere_cap(xx,angle);

        for (i=1;i<=4;i++)         /* angle1: bet 1 & 2; angle2: bet 2 & 3 ..*/
            E->sphere.angle[m][i] = angle[i];

        E->sphere.area[m] = area_sphere_cap(angle);

        for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)
            for (es=1;es<=E->lmesh.SNEL[lev];es++)              {
                el = (es-1)*E->lmesh.ELZ[lev]+1;
                for (i=1;i<=4;i++)
                    ia[i] = E->IEN[lev][m][el].node[i];

                for (i=1;i<=4;i++)  {
                    xx[1][i] = E->X[lev][m][1][ia[i]]/E->SX[lev][m][3][ia[1]];
                    xx[2][i] = E->X[lev][m][2][ia[i]]/E->SX[lev][m][3][ia[1]];
                    xx[3][i] = E->X[lev][m][3][ia[i]]/E->SX[lev][m][3][ia[1]];
                }

                get_angle_sphere_cap(xx,angle);

                for (i=1;i<=4;i++)         /* angle1: bet 1 & 2; angle2: bet 2 & 3 ..*/
                    E->sphere.angle1[lev][m][i][es] = angle[i];

                E->sphere.area1[lev][m][es] = area_sphere_cap(angle);

/*              fprintf(E->fp_out,"lev%d %d %.6e %.6e %.6e %.6e %.6e\n",lev,es,angle[1],angle[2],angle[3],angle[4],E->sphere.area1[lev][m][es]); */

            }  /* end for lev and es */

    }  /* end for m */

    return;
}

/* ================================================
   area of spherical rectangle
   ================================================ */
double area_sphere_cap(double angle[6])
{

    double area,a,b,c;

    a = angle[1];
    b = angle[2];
    c = angle[5];
    area = area_of_sphere_triag(a,b,c);

    a = angle[3];
    b = angle[4];
    c = angle[5];
    area += area_of_sphere_triag(a,b,c);

    return (area);
}

/* ================================================
   area of spherical triangle
   ================================================ */
double area_of_sphere_triag(double a, double b, double c)
{

    double ss,ak,aa,bb,cc,area;
    const double e_16 = 1.0e-16;
    const double two = 2.0;
    const double pt5 = 0.5;

    ss = (a+b+c)*pt5;
    area=0.0;
    a = sin(ss-a);
    b = sin(ss-b);
    c = sin(ss-c);
    ak = a*b*c/sin(ss);   /* sin(ss-a)*sin(ss-b)*sin(ss-c)/sin(ss)  */
    if(ak<e_16) return (area);
    ak = sqrt(ak);
    aa = two*atan(ak/a);
    bb = two*atan(ak/b);
    cc = two*atan(ak/c);
    area = aa+bb+cc-M_PI;

    return (area);
}

/*  =====================================================================
    get the area for given five points (4 nodes for a rectangle and one test node)
    angle [i]: angle bet test node and node i of the rectangle
    angle1[i]: angle bet nodes i and i+1 of the rectangle
    ====================================================================== */
double area_of_5points(struct All_variables *E, int lev, int m, int el, double x[4], int ne)
{
    int i,es,ia[5];
    double area1;
    double xx[4],angle[5],angle1[5];

    for (i=1;i<=4;i++)
        ia[i] = E->IEN[lev][m][el].node[i];

    es = (el-1)/E->lmesh.ELZ[lev]+1;

    for (i=1;i<=4;i++)                 {
        xx[1] = E->X[lev][m][1][ia[i]]/E->SX[lev][m][3][ia[1]];
        xx[2] = E->X[lev][m][2][ia[i]]/E->SX[lev][m][3][ia[1]];
        xx[3] = E->X[lev][m][3][ia[i]]/E->SX[lev][m][3][ia[1]];
        angle[i] = get_angle(x,xx);  /* get angle bet (i,j) and other four*/
        angle1[i]= E->sphere.angle1[lev][m][i][es];
    }

    area1 = area_of_sphere_triag(angle[1],angle[2],angle1[1])
        + area_of_sphere_triag(angle[2],angle[3],angle1[2])
        + area_of_sphere_triag(angle[3],angle[4],angle1[3])
        + area_of_sphere_triag(angle[4],angle[1],angle1[4]);

    return (area1);
}

/*  ================================
    get the angle for given four points spherical rectangle
    ================================= */

void  get_angle_sphere_cap(double xx[4][5], double angle[6])
{

    int i,j,ii;
    double y1[4],y2[4];

    for (i=1;i<=4;i++)     {     /* angle1: bet 1 & 2; angle2: bet 2 & 3 ..*/
        for (j=1;j<=3;j++)     {
            ii=(i==4)?1:(i+1);
            y1[j] = xx[j][i];
            y2[j] = xx[j][ii];
        }
        angle[i] = get_angle(y1,y2);
    }

    for (j=1;j<=3;j++) {
        y1[j] = xx[j][1];
        y2[j] = xx[j][3];
    }

    angle[5] = get_angle(y1,y2);     /* angle5 for betw 1 and 3: diagonal */
    return;
}

/*  ================================
    get the angle for given two points
    ================================= */
double get_angle(double x[4], double xx[4])
{
    double dist,angle;
    const double pt5 = 0.5;
    const double two = 2.0;

    dist=sqrt( (x[1]-xx[1])*(x[1]-xx[1])
               + (x[2]-xx[2])*(x[2]-xx[2])
               + (x[3]-xx[3])*(x[3]-xx[3]) )*pt5;
    angle = asin(dist)*two;

    return (angle);
}
