/* Functions relating to the building and use of mesh locations ... */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h>

/*   ======================================================================
     ======================================================================  */

void set_sphere_harmonics(E)
     struct All_variables *E;

{
    int m,node,ll,mm,i,j;
    static void compute_sphereh_table();

    i=0;
    for (ll=0;ll<=E->output.llmax;ll++)
	for (mm=0;mm<=ll;mm++)   {
	    E->sphere.hindex[ll][mm] = i;
	    i++;
	}

    E->sphere.hindice = i;

    /* spherical harmonic coeff (0=cos, 1=sin)
       for surface topo, cmb topo and geoid */
    for (i=0;i<=1;i++)   {
	E->sphere.harm_tpgt[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
	E->sphere.harm_tpgb[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
	E->sphere.harm_geoid[i]=(float*)malloc((E->sphere.hindice+2)*sizeof(float));
    }

    compute_sphereh_table(E);

    return;
}

/* =========================================================
   expand the field TG into spherical harmonics
   ========================================================= */
void sphere_expansion(E,TG,sphc,sphs)
     struct All_variables *E;
     float **TG,*sphc,*sphs;
{
    int el,nint,d,p,i,m,j,es,mm,ll,rand();
    //double t,f,sphere_h();
    void sum_across_surf_sph1();
    void get_global_1d_shape_fn();
    struct Shape_function1 M;
    struct Shape_function1_dA dGamma;

    for (i=0;i<E->sphere.hindice;i++)    {
	sphc[i] = 0.0;
	sphs[i] = 0.0;
    }

    for (m=1;m<=E->sphere.caps_per_proc;m++)
	for (es=1;es<=E->lmesh.snel;es++)   {
	    el = es*E->lmesh.elz;

	    get_global_1d_shape_fn(E,el,&M,&dGamma,1,m);

	    for (ll=0;ll<=E->output.llmax;ll++)
		for (mm=0; mm<=ll; mm++)   {

		    p = E->sphere.hindex[ll][mm];

		    for(nint=1;nint<=onedvpoints[E->mesh.nsd];nint++)   {
			for(d=1;d<=onedvpoints[E->mesh.nsd];d++)   {
			    j = E->sien[m][es].node[d];
			    sphc[p] += TG[m][E->sien[m][es].node[d]]
				* E->sphere.tablesplm[m][j][p]
				* E->sphere.tablescosf[m][j][mm]
				* E->M.vpt[GMVINDEX(d,nint)]
				* dGamma.vpt[GMVGAMMA(1,nint)];
			    sphs[p] += TG[m][E->sien[m][es].node[d]]
				* E->sphere.tablesplm[m][j][p]
				* E->sphere.tablessinf[m][j][mm]
				* E->M.vpt[GMVINDEX(d,nint)]
				* dGamma.vpt[GMVGAMMA(1,nint)];
			}
		    }

		}       /* end for ll and mm  */

	}

    sum_across_surf_sph1(E,sphc,sphs);

    return;
}


/* ==================================================*/
/* ==================================================*/
static void  compute_sphereh_table(E)
     struct All_variables *E;
{

    int m,node,ll,mm,i,j,p;
    double t,f;


    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
	E->sphere.tablesplm[m]   = (double **) malloc((E->lmesh.nsf+1)*sizeof(double*));
	E->sphere.tablescosf[m] = (double **) malloc((E->lmesh.nsf+1)*sizeof(double*));
	E->sphere.tablessinf[m] = (double **) malloc((E->lmesh.nsf+1)*sizeof(double*));

	for (i=1;i<=E->lmesh.nsf;i++)   {
	    E->sphere.tablesplm[m][i]= (double *)malloc((E->sphere.hindice+3)*sizeof(double));
	    E->sphere.tablescosf[m][i]= (double *)malloc((E->output.llmax+3)*sizeof(double));
	    E->sphere.tablessinf[m][i]= (double *)malloc((E->output.llmax+3)*sizeof(double));
	}
    }

    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
	for (j=1;j<=E->lmesh.nsf;j++)  {
	    node = j*E->lmesh.noz;
	    f=E->sx[m][2][node];
	    t=E->sx[m][1][node];
	    for (mm=0;mm<=E->output.llmax;mm++)   {
		E->sphere.tablescosf[m][j][mm] = cos( (double)(mm)*f );
		E->sphere.tablessinf[m][j][mm] = sin( (double)(mm)*f );
	    }

	    for (ll=0;ll<=E->output.llmax;ll++)
		for (mm=0;mm<=ll;mm++)  {
		    p = E->sphere.hindex[ll][mm];
		    E->sphere.tablesplm[m][j][p] = modified_plgndr_a(ll,mm,t) ;
		}
	}
    }

    return;
}

/* ================================================
   compute angle and area
   ================================================*/

void compute_angle_surf_area (E)
     struct All_variables *E;
{

    int es,el,m,i,j,ii,ia[5],lev;
    double aa,y1[4],y2[4],angle[6],xx[4][5],area_sphere_cap();
    void get_angle_sphere_cap();
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

		fprintf(E->fp_out,"lev%d %d %.6e %.6e %.6e %.6e %.6e\n",lev,es,angle[1],angle[2],angle[3],angle[4],E->sphere.area1[lev][m][es]);

	    }  /* end for lev and es */

    }  /* end for m */

    return;
}

/* ================================================
   area of spherical rectangle
   ================================================ */
double area_sphere_cap(angle)
     double angle[6];
{

    double area,a,b,c;
    double area_of_sphere_triag();

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
double area_of_sphere_triag(a,b,c)
     double a,b,c;
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
double area_of_5points(E,lev,m,el,x,ne)
     struct All_variables *E;
     int lev,m,el,ne;
     double x[4];
{
    int i,es,ia[5];
    double area1,get_angle(),area_of_sphere_triag();
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

void  get_angle_sphere_cap(xx,angle)
     double xx[4][5],angle[6];
{

    int i,j,ii;
    double y1[4],y2[4],get_angle();;

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
double get_angle(x,xx)
     double x[4],xx[4];
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


#if 0



#endif
