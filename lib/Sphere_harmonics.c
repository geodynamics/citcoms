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
    double modified_plgndr_a();

    int m,node,ll,mm,i,j,p;
    double t,f;


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

