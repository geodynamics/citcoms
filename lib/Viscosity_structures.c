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
/* Functions relating to the determination of viscosity field either
   as a function of the run, as an initial condition or as specified from
   a previous file */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"

static void apply_low_visc_wedge_channel(struct All_variables *E, float **evisc);
static void low_viscosity_channel_factor(struct All_variables *E, float *F);
static void low_viscosity_wedge_factor(struct All_variables *E, float *F);


void viscosity_system_input(struct All_variables *E)
{
    int m=E->parallel.me;
    int i;

    /* default values .... */
    for(i=0;i<40;i++) {
        E->viscosity.N0[i]=1.0;
        E->viscosity.T[i] = 0.0;
        E->viscosity.Z[i] = 0.0;
        E->viscosity.E[i] = 0.0;
    }

    /* read in information */
    input_boolean("VISC_UPDATE",&(E->viscosity.update_allowed),"on",m);
    input_int("rheol",&(E->viscosity.RHEOL),"3",m);
    input_int("num_mat",&(E->viscosity.num_mat),"1",m);
    input_float_vector("visc0",E->viscosity.num_mat,(E->viscosity.N0),m);

    input_boolean("TDEPV",&(E->viscosity.TDEPV),"on",m);
    if (E->viscosity.TDEPV) {
        input_float_vector("viscT",E->viscosity.num_mat,(E->viscosity.T),m);
        input_float_vector("viscE",E->viscosity.num_mat,(E->viscosity.E),m);
        input_float_vector("viscZ",E->viscosity.num_mat,(E->viscosity.Z),m);
    }


    E->viscosity.sdepv_misfit = 1.0;
    input_boolean("SDEPV",&(E->viscosity.SDEPV),"off",m);
    if (E->viscosity.SDEPV) {
        input_float("sdepv_misfit",&(E->viscosity.sdepv_misfit),"0.001",m);
        input_float_vector("sdepv_expt",E->viscosity.num_mat,(E->viscosity.sdepv_expt),m);
    }

    input_boolean("low_visc_channel",&(E->viscosity.channel),"off",m);
    input_boolean("low_visc_wedge",&(E->viscosity.wedge),"off",m);

    input_float("lv_min_radius",&(E->viscosity.lv_min_radius),"0.9764",m);
    input_float("lv_max_radius",&(E->viscosity.lv_max_radius),"0.9921",m);
    input_float("lv_channel_thickness",&(E->viscosity.lv_channel_thickness),"0.0047",m);
    input_float("lv_reduction",&(E->viscosity.lv_reduction),"0.5",m);

    input_boolean("VMAX",&(E->viscosity.MAX),"off",m);
    if (E->viscosity.MAX)
        input_float("visc_max",&(E->viscosity.max_value),"1e22,1,nomax",m);

    input_boolean("VMIN",&(E->viscosity.MIN),"off",m);
    if (E->viscosity.MIN)
        input_float("visc_min",&(E->viscosity.min_value),"1e20",m);

    return;
}


void viscosity_input(struct All_variables *E)
{
    int m = E->parallel.me;

    input_string("Viscosity",E->viscosity.STRUCTURE,NULL,m);
    input_int ("visc_smooth_method",&(E->viscosity.smooth_cycles),"0",m);

    if ( strcmp(E->viscosity.STRUCTURE,"system") == 0)
        E->viscosity.FROM_SYSTEM = 1;
    else
        E->viscosity.FROM_SYSTEM = 0;

    if (E->viscosity.FROM_SYSTEM)
        viscosity_system_input(E);

    return;
}



/* ============================================ */

void get_system_viscosity(E,propogate,evisc,visc)
     struct All_variables *E;
     int propogate;
     float **evisc,**visc;
{
    void visc_from_mat();
    void visc_from_T();
    void visc_from_S();
    void apply_viscosity_smoother();
    void visc_to_node_interpolate();
    void visc_from_nodes_to_gint();
    void visc_from_gint_to_nodes();


    int i,j,m;
    float temp1,temp2,*vvvis;
    double *TG;

    const int vpts = vpoints[E->mesh.nsd];

    if(E->viscosity.TDEPV)
        visc_from_T(E,evisc,propogate);
    else
        visc_from_mat(E,evisc);

    if(E->viscosity.SDEPV)
        visc_from_S(E,evisc,propogate);


    apply_low_visc_wedge_channel(E, evisc);


    /* min/max cut-off */

    if(E->viscosity.MAX) {
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.nel;i++)
                for(j=1;j<=vpts;j++)
                    if(evisc[m][(i-1)*vpts + j] > E->viscosity.max_value)
                        evisc[m][(i-1)*vpts + j] = E->viscosity.max_value;
    }

    if(E->viscosity.MIN) {
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.nel;i++)
                for(j=1;j<=vpts;j++)
                    if(evisc[m][(i-1)*vpts + j] < E->viscosity.min_value)
                        evisc[m][(i-1)*vpts + j] = E->viscosity.min_value;
    }

    /*
      if (E->control.verbose)  {
      fprintf(E->fp_out,"output_evisc \n");
      for(m=1;m<=E->sphere.caps_per_proc;m++) {
      fprintf(E->fp_out,"output_evisc for cap %d\n",E->sphere.capid[m]);
      for(i=1;i<=E->lmesh.nel;i++)
      fprintf(E->fp_out,"%d %d %f %f\n",i,E->mat[m][i],evisc[m][(i-1)*vpts+1],evisc[m][(i-1)*vpts+7]);
      }
      fflush(E->fp_out);
      }
    */

    visc_from_gint_to_nodes(E,evisc,visc,E->mesh.levmax);

    visc_from_nodes_to_gint(E,visc,evisc,E->mesh.levmax);

    /*    visc_to_node_interpolate(E,evisc,visc);
     */

    /*    for(m=1;m<=E->sphere.caps_per_proc;m++) {
          for(i=1;i<=E->lmesh.nel;i++)
          if (i%E->lmesh.elz==0) {
          fprintf(E->fp_out,"%.4e %.4e %.4e %5d %2d\n",E->eco[m][i].centre[1],E->eco[m][i].centre[2],log10(evisc[m][(i-1)*vpts+1]),i,E->mat[m][i]);

	  }
          }  */
    return;
}



void initial_viscosity(struct All_variables *E)
{
    if (E->viscosity.FROM_SYSTEM)
        get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);

    return;
}


void visc_from_mat(E,EEta)
     struct All_variables *E;
     float **EEta;
{

    int i,m,jj;

    for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(i=1;i<=E->lmesh.nel;i++)
            for(jj=1;jj<=vpoints[E->mesh.nsd];jj++)
                EEta[m][ (i-1)*vpoints[E->mesh.nsd]+jj ]=E->viscosity.N0[E->mat[m][i]-1];

    return;
}

void visc_from_T(E,EEta,propogate)
     struct All_variables *E;
     float **EEta;
     int propogate;
{
    int m,i,j,k,l,z,jj,kk,imark;
    float zero,e_6,one,eta0,Tave,depth,temp,tempa,temp1,TT[9];
    float zzz,zz[9];
    float visc1, visc2, tempa_exp;
    const int vpts = vpoints[E->mesh.nsd];
    const int ends = enodes[E->mesh.nsd];
    const int nel = E->lmesh.nel;

    e_6 = 1.e-6;
    one = 1.0;
    zero = 0.0;
    imark = 0;

    switch (E->viscosity.RHEOL)   {
    case 1:
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i];

                if(E->control.mat_control==0)
                    tempa = E->viscosity.N0[l-1];
                else if(E->control.mat_control==1)
                    tempa = E->viscosity.N0[l-1]*E->VIP[m][i];

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    EEta[m][ (i-1)*vpts + jj ] = tempa*
                        exp( E->viscosity.E[l-1] * (E->viscosity.T[l-1] - temp));

                }
            }
        break;

    case 2:
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i];

                if(E->control.mat_control==0)
                    tempa = E->viscosity.N0[l-1];
                else if(E->control.mat_control==1)
                    tempa = E->viscosity.N0[l-1]*E->VIP[m][i];

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    EEta[m][ (i-1)*vpts + jj ] = tempa*
                        exp( -temp / E->viscosity.T[l-1]);

                }
            }
        break;

    case 3:

        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i];
                tempa = E->viscosity.N0[l-1];
                j = 0;

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                    zz[kk] = (1.-E->sx[m][3][E->ien[m][i].node[kk]]);
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        TT[kk]=max(TT[kk],zero);
                        temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
                        zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    if(E->control.mat_control==0)
                        EEta[m][ (i-1)*vpts + jj ] = tempa*
                            exp( E->viscosity.E[l-1]/(temp+E->viscosity.T[l-1])
                                 - E->viscosity.E[l-1]/(one +E->viscosity.T[l-1]) );

                    if(E->control.mat_control==1)
                        EEta[m][ (i-1)*vpts + jj ] = tempa*E->VIP[m][i]*
                            exp( E->viscosity.E[l-1]/(temp+E->viscosity.T[l-1])
                                 - E->viscosity.E[l-1]/(one +E->viscosity.T[l-1]) );
                }
            }
        break;

    case 4:

        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i];
                tempa = E->viscosity.N0[l-1];
                j = 0;

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                    zz[kk] = (1.-E->sx[m][3][E->ien[m][i].node[kk]]);
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        TT[kk]=max(TT[kk],zero);
                        temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
                        zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    /* The viscosity formulation (dimensional) is: visc=visc0*exp[(Ea+p*Va)/R*T]
                       Typical values for dry upper mantle are: Ea = 300 KJ/mol ; Va = 1.e-5 m^3/mol
                       T=T0+DT*T'; where DT - temperature contrast (from Rayleigh number)
                       T' - nondimensional temperature; T0 - surface tempereture (273 K)
                       T=DT*[(T0/DT) + T'] => visc=visc0*exp{(Ea+p*Va)/R*DT*[(T0/DT) + T']}
                       visc=visc0*exp{[(Ea/R*DT) + (p*Va/R*DT)]/[(T0/DT) + T']}
                       so: E->viscosity.E = Ea/R*DT ; E->viscosity.Z = Va/R*DT
                       p = zzz and E->viscosity.T = T0/DT */


                    if(E->control.mat_control==0)
                        EEta[m][ (i-1)*vpts + jj ] = tempa*
                            exp( (E->viscosity.E[l-1] +  E->viscosity.Z[l-1]*zzz )
                                 / (E->viscosity.T[l-1]+temp) );



                    if(E->control.mat_control==1)
                        EEta[m][ (i-1)*vpts + jj ] = tempa*E->VIP[m][i]*
                            exp( (E->viscosity.E[l-1] +  E->viscosity.Z[l-1]*zzz )
                                 / (E->viscosity.T[l-1]+temp) );

                }
            }
        break;


    case 5:

        /* same as rheol 3, except alternative margin, VIP, formulation */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i];
                tempa = E->viscosity.N0[l-1];
                j = 0;

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                    zz[kk] = (1.-E->sx[m][3][E->ien[m][i].node[kk]]);
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        TT[kk]=max(TT[kk],zero);
                        temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
                        zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    if(E->control.mat_control==0)
                        EEta[m][ (i-1)*vpts + jj ] = tempa*
                            exp( E->viscosity.E[l-1]/(temp+E->viscosity.T[l-1])
                                 - E->viscosity.E[l-1]/(one +E->viscosity.T[l-1]) );

                    if(E->control.mat_control==1) {
                     /* visc1 = E->VIP[m][i];
                        visc2 = 2.0/(1./visc1 + 1.);
                        tempa_exp = tempa*
	                exp( E->viscosity.E[l-1]/(temp+E->viscosity.T[l-1])
		           - E->viscosity.E[l-1]/(one +E->viscosity.T[l-1]) );
                        visc1 = tempa*E->viscosity.max_value;
                        if(tempa_exp > visc1) tempa_exp=visc1;
                        EEta[m][ (i-1)*vpts + jj ] = visc2*tempa_exp;
                     */
                       visc1 = E->VIP[m][i];
                       visc2 = tempa*
	               exp( E->viscosity.E[l-1]/(temp+E->viscosity.T[l-1])
		          - E->viscosity.E[l-1]/(one +E->viscosity.T[l-1]) );
                       if(visc1 <= 0.95) visc2=visc1;
                       EEta[m][ (i-1)*vpts + jj ] = visc2;
                      }

                }
            }
        break;




    }

    return;
}


void visc_from_S(E,EEta,propogate)
     struct All_variables *E;
     float **EEta;
     int propogate;
{
    float one,two,scale,stress_magnitude,depth,exponent1;
    float *eedot;

    void strain_rate_2_inv();
    int m,e,l,z,jj,kk;

    const int vpts = vpoints[E->mesh.nsd];
    const int nel = E->lmesh.nel;

    eedot = (float *) malloc((2+nel)*sizeof(float));
    one = 1.0;
    two = 2.0;

    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
        strain_rate_2_inv(E,m,eedot,1);

        for(e=1;e<=nel;e++)   {
            exponent1= one/E->viscosity.sdepv_expt[E->mat[m][e]-1];
            scale=pow(eedot[e],exponent1-one);
            for(jj=1;jj<=vpts;jj++)
                EEta[m][(e-1)*vpts + jj] = scale*pow(EEta[m][(e-1)*vpts+jj],exponent1);
        }
    }

    free ((void *)eedot);
    return;
}



void strain_rate_2_inv(E,m,EEDOT,SQRT)
     struct All_variables *E;
     float *EEDOT;
     int m,SQRT;
{
    void get_global_shape_fn();
    void velo_from_element();

    struct Shape_function GN;
    struct Shape_function_dA dOmega;
    struct Shape_function_dx GNx;



    double edot[4][4], rtf[4][9];
    float VV[4][9], Vxyz[7][9], dilation[9];

    int e, i, j, p, q, n;

    const int nel = E->lmesh.nel;
    const int dims = E->mesh.nsd;
    const int ends = enodes[dims];
    const int lev = E->mesh.levmax;
    const int ppts = ppoints[dims];
    const int sphere_key = 1;


    for(e=1; e<=nel; e++) {

        /* get shape function on presure gauss points */
        get_global_shape_fn(E, e, &GN, &GNx, &dOmega, 2,
                                sphere_key, rtf, lev, m);

        velo_from_element(E, VV, m, e, sphere_key);


        /* Vxyz is the strain rate vector, whose relationship with
         * the strain rate tensor (e) is that:
         *    Vxyz[1] = e11
         *    Vxyz[2] = e22
         *    Vxyz[3] = e33
         *    Vxyz[4] = 2*e12
         *    Vxyz[5] = 2*e13
         *    Vxyz[6] = 2*e23
         * where 1 is theta, 2 is phi, and 3 is r
         */
        for(j=1; j<=ppts; j++) {
            Vxyz[1][j] = 0.0;
            Vxyz[2][j] = 0.0;
            Vxyz[3][j] = 0.0;
            Vxyz[4][j] = 0.0;
            Vxyz[5][j] = 0.0;
            Vxyz[6][j] = 0.0;
        }

        for(j=1; j<=ppts; j++) {
            for(i=1; i<=ends; i++) {
                Vxyz[1][j] += (VV[1][i] * GNx.ppt[GNPXINDEX(0, i, j)]
                               + VV[3][i] * E->N.ppt[GNPINDEX(i, j)])
                    * rtf[3][j];
                Vxyz[2][j] += ((VV[2][i] * GNx.ppt[GNPXINDEX(1, i, j)]
                                + VV[1][i] * E->N.ppt[GNPINDEX(i, j)]
                                * cos(rtf[1][j])) / sin(rtf[1][j])
                               + VV[3][i] * E->N.ppt[GNPINDEX(i, j)])
                    * rtf[3][j];
                Vxyz[3][j] += VV[3][i] * GNx.ppt[GNPXINDEX(2, i, j)];

                Vxyz[4][j] += ((VV[1][i] * GNx.ppt[GNPXINDEX(1, i, j)]
                                - VV[2][i] * E->N.ppt[GNPINDEX(i, j)]
                                * cos(rtf[1][j])) / sin(rtf[1][j])
                               + VV[2][i] * GNx.ppt[GNPXINDEX(0, i, j)])
                    * rtf[3][j];
                Vxyz[5][j] += VV[1][i] * GNx.ppt[GNPXINDEX(2, i, j)]
                    + rtf[3][j] * (VV[3][i] * GNx.ppt[GNPXINDEX(0, i, j)]
                                   - VV[1][i] * E->N.ppt[GNPINDEX(i, j)]);
                Vxyz[6][j] += VV[2][i] * GNx.ppt[GNPXINDEX(2, i, j)]
                    + rtf[3][j] * (VV[3][i]
                                   * GNx.ppt[GNPXINDEX(1, i, j)]
                                   / sin(rtf[1][j])
                                   - VV[2][i] * E->N.ppt[GNPINDEX(i, j)]);
            }
        }

        if(E->control.inv_gruneisen > 0)
            for(j=1; j<=ppts; j++)
                dilation[j] = (Vxyz[1][j] + Vxyz[2][j] + Vxyz[3][j]) / 3.0;
        else
            for(j=1; j<=ppts; j++)
                dilation[j] = 0;

        edot[1][1] = edot[2][2] = edot[3][3] = 0;
        edot[1][2] = edot[1][3] = edot[2][3] = 0;

        /* edot is 2 * (the deviatoric strain rate tensor) */
        for(j=1; j<=ppts; j++) {
            edot[1][1] += 2.0 * (Vxyz[1][j] - dilation[j]);
            edot[2][2] += 2.0 * (Vxyz[2][j] - dilation[j]);
            edot[3][3] += 2.0 * (Vxyz[3][j] - dilation[j]);
            edot[1][2] += Vxyz[4][j];
            edot[1][3] += Vxyz[5][j];
            edot[2][3] += Vxyz[6][j];
        }

        /* TODO: figure out why 2.0 factor is here */
        EEDOT[e] = edot[1][1] * edot[1][1]
            + edot[1][2] * edot[1][2] * 2.0
            + edot[2][2] * edot[2][2]
            + edot[2][3] * edot[2][3] * 2.0
            + edot[3][3] * edot[3][3]
            + edot[1][3] * edot[1][3] * 2.0;
    }

    if(SQRT)
	for(e=1;e<=nel;e++)
	    EEDOT[e] =  sqrt(0.5 *EEDOT[e]);
    else
	for(e=1;e<=nel;e++)
	    EEDOT[e] *=  0.5;

    return;
}



void visc_to_node_interpolate(E,evisc,visc)
     struct All_variables *E;
     float **evisc,**visc;
{

    /*  void exchange_node_f(); */
    /*  void get_global_shape_fn(); */
    /*  void return_horiz_ave_f(); */
    /*  void sphere_interpolate(); */
    /*  void print_interpolated(); */
    /*  void gather_TG_to_me0(); */
    /*  void parallel_process_termination(); */
    /*  int i,j,k,e,node,snode,m,nel2; */
    /*    FILE *fp; */
    /*    char output_file[255]; */

    /*  float *TG,t,f,rad, Szz; */

    /*  double time1,CPU_time0(),tww[9],rtf[4][9]; */

    /*  struct Shape_function GN; */
    /*  struct Shape_function_dA dOmega; */
    /*  struct Shape_function_dx GNx; */

    /*  const int dims=E->mesh.nsd,dofs=E->mesh.dof; */
    /*  const int vpts=vpoints[dims]; */
    /*  const int ppts=ppoints[dims]; */
    /*  const int ends=enodes[dims]; */
    /*  const int nno=E->lmesh.nno; */
    /*  const int lev=E->mesh.levmax; */


    /*     TG =(float *)malloc((E->sphere.nsf+1)*sizeof(float)); */
    /*     for (i=E->sphere.nox;i>=1;i--) */
    /*       for (j=1;j<=E->sphere.noy;j++)  { */
    /*            node = i + (j-1)*E->sphere.nox; */
    /* 	   TG[node] = 0.0; */
    /*   	   m = E->sphere.int_cap[node]; */
    /* 	   e = E->sphere.int_ele[node]; */

    /* 	   if (m>0 && e>0) { */
    /* 	      e=e+E->lmesh.elz-1; */
    /* 	      TG[node] = log10(evisc[m][(e-1)*vpts+1]); */
    /* 	      } */
    /* 	   } */

    /*     gather_TG_to_me0(E,TG); */

    /*     if (E->parallel.me==E->parallel.nprocz-1)  { */
    /*      sprintf(output_file,"%s.evisc_intp",E->control.data_file); */
    /*      fp=fopen(output_file,"w"); */

    /*     rad = 180/M_PI; */
    /*     for (i=E->sphere.nox;i>=1;i--) */
    /*       for (j=1;j<=E->sphere.noy;j++)  { */
    /*            node = i + (j-1)*E->sphere.nox; */
    /*            t = 90-E->sphere.sx[1][node]*rad; */
    /* 	   f = E->sphere.sx[2][node]*rad; */
    /* 	   fprintf (fp,"%.3e %.3e %.4e\n",f,t,TG[node]); */
    /* 	   } */
    /*       fclose(fp); */
    /*      } */

    /*  free((void *)TG); */

    return;
}


static void apply_low_visc_wedge_channel(struct All_variables *E, float **evisc)
{
    int i,j,jj,k,m,n,mm,ii,nn;
    float temp1,temp2,*vvvis;

    float THETA_LOCAL_ELEM_T, THETA_LOCAL_ELEM, FI_LOCAL_ELEM_T, FI_LOCAL_ELEM, R_LOCAL_ELEM_T, R_LOCAL_ELEM;
    const int vpts = vpoints[E->mesh.nsd];

    float *F;

    F = (float *)malloc((E->lmesh.nel+1)*sizeof(float));
    for(i=1 ; i<=E->lmesh.nel ; i++)
        F[i] = 0.0;

    /* if low viscosity channel ... */
    if(E->viscosity.channel)
        low_viscosity_channel_factor(E, F);


    /* if low viscosity wedge ... */
    if(E->viscosity.wedge)
        low_viscosity_wedge_factor(E, F);


    for(i=1 ; i<=E->lmesh.nel ; i++) {
        if (F[i] != 0.0)
            for(m = 1 ; m <= E->sphere.caps_per_proc ; m++) {
                for(j=1;j<=vpts;j++) {
                    evisc[m][(i-1)*vpts + j] = F[i];
            }
        }
    }


    free(F);

    return;
}




static void low_viscosity_channel_factor(struct All_variables *E, float *F)
{
    int i, n;
    float THETA_LOCAL_ELEM_T, THETA_LOCAL_ELEM, FI_LOCAL_ELEM_T, FI_LOCAL_ELEM, R_LOCAL_ELEM_T, R_LOCAL_ELEM;


    for(i=1 ; i<=E->lmesh.nel ; i++) {
        THETA_LOCAL_ELEM = E->Tracer.THETA_LOC_ELEM[i];
        FI_LOCAL_ELEM = E->Tracer.FI_LOC_ELEM[i];
        R_LOCAL_ELEM = E->Tracer.R_LOC_ELEM[i];

        if (R_LOCAL_ELEM >= E->viscosity.lv_min_radius && R_LOCAL_ELEM <= E->viscosity.lv_max_radius) {

            for(n=1 ; n<=E->Tracer.LOCAL_NUM_TRACERS ; n++) {
                if (E->Tracer.tracer_z[n] <= E->viscosity.lv_max_radius && E->Tracer.tracer_z[n] >= E->viscosity.lv_min_radius) {
                    THETA_LOCAL_ELEM_T = E->Tracer.THETA_LOC_ELEM_T[n];
                    FI_LOCAL_ELEM_T = E->Tracer.FI_LOC_ELEM_T[n];
                    R_LOCAL_ELEM_T = E->Tracer.R_LOC_ELEM_T[n];

                    if((R_LOCAL_ELEM >= R_LOCAL_ELEM_T) &&
                       (R_LOCAL_ELEM <= R_LOCAL_ELEM_T+E->viscosity.lv_channel_thickness) &&
                       (FI_LOCAL_ELEM == FI_LOCAL_ELEM_T) &&
                       (THETA_LOCAL_ELEM == THETA_LOCAL_ELEM_T)) {

                        F[i] = E->viscosity.lv_reduction;

                    }
                }

            }
        }
    }

}



static void low_viscosity_wedge_factor(struct All_variables *E, float *F)
{
    int i, n;
    float THETA_LOCAL_ELEM_T, THETA_LOCAL_ELEM, FI_LOCAL_ELEM_T, FI_LOCAL_ELEM, R_LOCAL_ELEM_T, R_LOCAL_ELEM;

    for(i=1 ; i<=E->lmesh.nel ; i++) {
        THETA_LOCAL_ELEM = E->Tracer.THETA_LOC_ELEM[i];
        FI_LOCAL_ELEM = E->Tracer.FI_LOC_ELEM[i];
        R_LOCAL_ELEM = E->Tracer.R_LOC_ELEM[i];

        if (R_LOCAL_ELEM <= E->viscosity.lv_max_radius && R_LOCAL_ELEM >= E->viscosity.lv_min_radius) {

            for(n=1 ; n<=E->Tracer.LOCAL_NUM_TRACERS ; n++) {
                if (E->Tracer.tracer_z[n] <= E->viscosity.lv_max_radius && E->Tracer.tracer_z[n] >= E->viscosity.lv_min_radius) {
                    THETA_LOCAL_ELEM_T = E->Tracer.THETA_LOC_ELEM_T[n];
                    FI_LOCAL_ELEM_T = E->Tracer.FI_LOC_ELEM_T[n];
                    R_LOCAL_ELEM_T = E->Tracer.R_LOC_ELEM_T[n];


                    if((R_LOCAL_ELEM >= R_LOCAL_ELEM_T) &&
                       (FI_LOCAL_ELEM == FI_LOCAL_ELEM_T) &&
                       (THETA_LOCAL_ELEM == THETA_LOCAL_ELEM_T)) {

                        F[i] = E->viscosity.lv_reduction;
                    }
                }
            }
        }
    }
}
