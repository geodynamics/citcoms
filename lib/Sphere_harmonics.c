/* Functions relating to the building and use of mesh locations ... */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h>

static void compute_sphereh_table(struct All_variables *);

/*   ======================================================================
     ======================================================================  */

void set_sphere_harmonics(E)
     struct All_variables *E;

{
    int m,node,ll,mm,i,j;

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
        E->sphere.harm_geoid[i]=(float*)malloc(E->sphere.hindice*sizeof(float));
        E->sphere.harm_geoid_from_bncy[i]=(float*)malloc(E->sphere.hindice*sizeof(float));
        E->sphere.harm_geoid_from_tpgt[i]=(float*)malloc(E->sphere.hindice*sizeof(float));
        E->sphere.harm_geoid_from_tpgb[i]=(float*)malloc(E->sphere.hindice*sizeof(float));

        E->sphere.harm_tpgt[i]=(float*)malloc(E->sphere.hindice*sizeof(float));
        E->sphere.harm_tpgb[i]=(float*)malloc(E->sphere.hindice*sizeof(float));
    }

    compute_sphereh_table(E);

    return;
}

/* =====================================
   Generalized Legendre polynomials
   =====================================*/
double modified_plgndr_a(int l, int m, double t)
{
    int i,ll;
    double x,fact1,fact2,fact,pll,pmm,pmmp1,somx2,plgndr;
    const double three=3.0;
    const double two=2.0;
    const double one=1.0;

    x = cos(t);
    pmm=one;
    if(m>0) {
        somx2=sqrt((one-x)*(one+x));
        fact1= three;
        fact2= two;
        for (i=1;i<=m;i++)   {
            fact=sqrt(fact1/fact2);
            pmm = -pmm*fact*somx2;
            fact1+=  two;
            fact2+=  two;
        }
    }

    if (l==m)
        plgndr = pmm;
    else  {
        pmmp1 = x*sqrt(two*m+three)*pmm;
        if(l==m+1)
            plgndr = pmmp1;
        else   {
            for (ll=m+2;ll<=l;ll++)  {
                fact1= sqrt((4.0*ll*ll-one)*(double)(ll-m)/(double)(ll+m));
                fact2= sqrt((2.0*ll+one)*(ll-m)*(ll+m-one)*(ll-m-one)
                            /(double)((two*ll-three)*(ll+m)));
                pll = ( x*fact1*pmmp1-fact2*pmm)/(ll-m);
                pmm = pmmp1;
                pmmp1 = pll;
            }
            plgndr = pll;
        }
    }

    plgndr /= sqrt(4.0*M_PI);

    if (m!=0) plgndr *= sqrt(two);

    return plgndr;
}


/* =========================================================
   expand the field TG into spherical harmonics
   ========================================================= */
void sphere_expansion(E,TG,sphc,sphs)
     struct All_variables *E;
     float **TG,*sphc,*sphs;
{
    int el,nint,d,p,i,m,j,es,mm,ll,rand();
    void sum_across_surf_sph1();

    for (i=0;i<E->sphere.hindice;i++)    {
        sphc[i] = 0.0;
        sphs[i] = 0.0;
    }

    for (m=1;m<=E->sphere.caps_per_proc;m++)
        for (es=1;es<=E->lmesh.snel;es++)   {

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
                                * E->surf_det[m][nint][es];
                            sphs[p] += TG[m][E->sien[m][es].node[d]]
                                * E->sphere.tablesplm[m][j][p]
                                * E->sphere.tablessinf[m][j][mm]
                                * E->M.vpt[GMVINDEX(d,nint)]
                                * E->surf_det[m][nint][es];
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
    double modified_plgndr_a();

    int m,node,ll,mm,i,j,p;
    double t,f,mmf;
    

    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
        E->sphere.tablesplm[m]   = (double **) malloc((E->lmesh.nsf+1)*sizeof(double*));
        E->sphere.tablescosf[m] = (double **) malloc((E->lmesh.nsf+1)*sizeof(double*));
        E->sphere.tablessinf[m] = (double **) malloc((E->lmesh.nsf+1)*sizeof(double*));

        for (i=1;i<=E->lmesh.nsf;i++)   {
            E->sphere.tablesplm[m][i]= (double *)malloc((E->sphere.hindice)*sizeof(double));
            E->sphere.tablescosf[m][i]= (double *)malloc((E->output.llmax+1)*sizeof(double));
            E->sphere.tablessinf[m][i]= (double *)malloc((E->output.llmax+1)*sizeof(double));
        }
    }

    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
        for (j=1;j<=E->lmesh.nsf;j++)  {
            node = j*E->lmesh.noz;
            f=E->sx[m][2][node];
            t=E->sx[m][1][node];
            for (mm=0;mm<=E->output.llmax;mm++)   {
	      mmf = (double)(mm)*f;
                E->sphere.tablescosf[m][j][mm] = cos( mmf );
                E->sphere.tablessinf[m][j][mm] = sin( mmf );
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

