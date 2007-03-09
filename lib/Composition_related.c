/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include <math.h>
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"


static void initialize_old_composition(struct All_variables *E);


void composition_input(struct All_variables *E)
{
    int m = E->parallel.me;

    input_int("chemical_buoyancy",&(E->composition.ichemical_buoyancy),
              "1,0,nomax",m);

    if (E->composition.ichemical_buoyancy==1) {

	input_double("buoyancy_ratio",
		     &(E->composition.buoyancy_ratio),"1.0",m);

	/* ibuoy_type=0 (absolute method) */
	/* ibuoy_type=1 (ratio method) */

	input_int("buoy_type",&(E->composition.ibuoy_type),"1,0,nomax",m);
	if (E->composition.ibuoy_type!=1) {
	    fprintf(stderr,"Terror-Sorry, only ratio method allowed now\n");
	    fflush(stderr);
	    parallel_process_termination();
	}

	if (E->trace.ic_method==2) {
	    input_int("reset_initial_composition",
		      &(E->composition.ireset_initial_composition),"0",m);
	}

        input_double("z_interface",&(E->composition.z_interface),"0.5",m);

    }


    /* compositional rheology */

    /* icompositional_rheology=0 (off) */
    /* icompositional_rheology=1 (on) */

    input_int("compositional_rheology",
	      &(E->composition.icompositional_rheology),"1,0,nomax",m);

    if (E->composition.icompositional_rheology==1) {
	input_double("compositional_prefactor",
		     &(E->composition.compositional_rheology_prefactor),
		     "1.0",m);
    }

}



void composition_setup(struct All_variables *E)
{

    E->composition.on = 0;
    if (E->composition.ichemical_buoyancy==1 ||
        E->composition.icompositional_rheology)
        E->composition.on = 1;

    if (E->composition.ichemical_buoyancy==1 && E->composition.ibuoy_type==1) {
	initialize_old_composition(E);
	fill_composition(E);
    }

}



void init_tracer_composition(struct All_variables *E)
{
    int j, kk, number_of_tracers;
    double rad;

    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        number_of_tracers = E->trace.itrac[j][0];
        for (kk=1;kk<=number_of_tracers;kk++) {
            rad = E->trace.rtrac[j][2][kk];

            if (rad<=E->composition.z_interface) E->trace.etrac[j][0][kk]=1.0;
            if (rad>E->composition.z_interface) E->trace.etrac[j][0][kk]=0.0;
        }
    }
    return;
}


void write_composition_instructions(struct All_variables *E)
{

    if (E->composition.ichemical_buoyancy==0)
	    fprintf(E->trace.fpt,"Passive Tracers\n");

    if (E->composition.ichemical_buoyancy==1)
        fprintf(E->trace.fpt,"Active Tracers\n");


    if (E->composition.ibuoy_type==1) fprintf(E->trace.fpt,"Ratio Method\n");
    if (E->composition.ibuoy_type==0) fprintf(E->trace.fpt,"Absolute Method\n");

    fprintf(E->trace.fpt,"Buoyancy Ratio: %f\n", E->composition.buoyancy_ratio);


    if (E->composition.ireset_initial_composition==0)
	fprintf(E->trace.fpt,"Using old initial composition from tracer files\n");
    else
	fprintf(E->trace.fpt,"Resetting initial composition\n");



    if (E->composition.icompositional_rheology==0) {
	fprintf(E->trace.fpt,"Compositional Rheology - OFF\n");
    }
    else if (E->composition.icompositional_rheology>0) {
	fprintf(E->trace.fpt,"Compositional Rheology - ON\n");
	fprintf(E->trace.fpt,"Compositional Prefactor: %f\n",
		E->composition.compositional_rheology_prefactor);
    }


    fflush(E->trace.fpt);
    fflush(stderr);

}


/************ FILL COMPOSITION ************************/
void fill_composition(struct All_variables *E)
{

    void compute_elemental_composition_ratio_method();
    void map_composition_to_nodes();

    /* Currently, only the ratio method works here.                */
    /* Will have to come back here to include the absolute method. */

    /* ratio method */

    if (E->composition.ibuoy_type==1)
	{
	    compute_elemental_composition_ratio_method(E);
	}

    /* absolute method */

    if (E->composition.ibuoy_type!=1)
	{
	    fprintf(E->trace.fpt,"Error(compute...)-only ratio method now\n");
	    fflush(E->trace.fpt);
	    exit(10);
	}

    /* Map elemental composition to nodal points */

    map_composition_to_nodes(E);

    return;
}



/************ INITIALIZE OLD COMPOSITION ************************/
static void initialize_old_composition(struct All_variables *E)
{

    char output_file[200];
    char input_s[1000];

    int ibottom_node;
    int kk;
    int j;

    double zbottom;
    double time;

    FILE *fp;

    for (j=1;j<=E->sphere.caps_per_proc;j++)
	{

	    if ((E->trace.oldel[j]=(double *)malloc((E->lmesh.nel+1)*sizeof(double)))==NULL)
		{
		    fprintf(E->trace.fpt,"ERROR(fill old composition)-no memory 324c\n");
		    fflush(E->trace.fpt);
		    exit(10);
		}
	}


    if ((E->trace.ic_method==0)||(E->trace.ic_method==1))
	{
	    for (j=1;j<=E->sphere.caps_per_proc;j++)
		{
		    for (kk=1;kk<=E->lmesh.nel;kk++)
			{

			    ibottom_node=E->ien[j][kk].node[1];
			    zbottom=E->sx[j][3][ibottom_node];

			    if (zbottom<E->composition.z_interface) E->trace.oldel[j][kk]=1.0;
			    if (zbottom>=E->composition.z_interface) E->trace.oldel[j][kk]=0.0;

			} /* end kk */
		} /* end j */
	}


    /* Else read from file */


    else if (E->trace.ic_method==2)
	{

	    /* first look for backing file */

	    sprintf(output_file,"%s.comp_el.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
	    if ( (fp=fopen(output_file,"r"))==NULL)
		{
                    fprintf(E->trace.fpt,"AKMerror(Initialize Old Composition)-FILE NOT EXIST: %s\n", output_file);
                    fflush(E->trace.fpt);
                    exit(10);
                }

            fgets(input_s,200,fp);

	    for(j=1;j<=E->sphere.caps_per_proc;j++)
		{
		    fgets(input_s,200,fp);
		    for (kk=1;kk<=E->lmesh.nel;kk++)
			{
			    fgets(input_s,200,fp);
			    sscanf(input_s,"%lf",&E->trace.oldel[j][kk]);
			}
		}

	    fclose(fp);

	} /* endif */



    return;
}



/*********** COMPUTE ELEMENTAL COMPOSITION RATIO METHOD ***/
/*                                                        */
/* This function computes the composition per element.    */
/* This function computes the composition per element.    */
/* Integer array ieltrac stores tracers per element.      */
/* Double array celtrac stores the sum of tracer composition */

void compute_elemental_composition_ratio_method(E)
     struct All_variables *E;
{

    int kk;
    int numtracers;
    int nelem;
    int j;
    int iempty=0;

    double comp;

    static int been_here=0;

    if (been_here==0)
	{
	    for (j=1;j<=E->sphere.caps_per_proc;j++)
		{

		    if ((E->trace.ieltrac[j]=(int *)malloc((E->lmesh.nel+1)*sizeof(int)))==NULL)
			{
			    fprintf(E->trace.fpt,"AKM(compute_elemental_composition)-no memory 5u83a\n");
			    fflush(E->trace.fpt);
			    exit(10);
			}
		    if ((E->trace.celtrac[j]=(double *)malloc((E->lmesh.nel+1)*sizeof(double)))==NULL)
			{
			    fprintf(E->trace.fpt,"AKM(compute_elemental_composition)-no memory 58hy8\n");
			    fflush(E->trace.fpt);
			    exit(10);
			}
		    if ((E->trace.comp_el[j]=(double *)malloc((E->lmesh.nel+1)*sizeof(double)))==NULL)
			{
			    fprintf(E->trace.fpt,"AKM(compute_elemental_composition)-no memory 8989y\n");
			    fflush(E->trace.fpt);
			    exit(10);
			}
		}

	    been_here++;

	}

    for (j=1;j<=E->sphere.caps_per_proc;j++)
	{

	    /* first zero arrays */

	    for (kk=1;kk<=E->lmesh.nel;kk++)
		{
		    E->trace.ieltrac[j][kk]=0;
		    E->trace.celtrac[j][kk]=0.0;
		}

	    numtracers=E->trace.itrac[j][0];

	    /* Fill ieltrac and celtrac */


	    for (kk=1;kk<=numtracers;kk++)
		{

		    nelem=E->trace.itrac[j][kk];
		    E->trace.ieltrac[j][nelem]++;

		    comp=E->trace.etrac[j][0][kk];

		    if (comp>1.0000001)
			{
			    fprintf(E->trace.fpt,"ERROR(compute elemental)-not ready for comp>1 yet (%f)(tr. %d) \n",comp,kk);
			    fflush(E->trace.fpt);
			    exit(10);
			}

		    E->trace.celtrac[j][nelem]=E->trace.celtrac[j][nelem]+comp;

		}

	    /* Check for empty entries and compute ratio.  */
	    /* If no tracers are in an element, use previous composition */

	    iempty=0;

	    for (kk=1;kk<=E->lmesh.nel;kk++)
		{

		    if (E->trace.ieltrac[j][kk]==0)
			{
			    iempty++;
			    E->trace.comp_el[j][kk]=E->trace.oldel[j][kk];
			}
		    else if (E->trace.ieltrac[j][kk]>0)
			{
			    E->trace.comp_el[j][kk]=E->trace.celtrac[j][kk]/(1.0*E->trace.ieltrac[j][kk]);
			}

		    if (E->trace.comp_el[j][kk]>(1.000001) || E->trace.comp_el[j][kk]<(-0.000001))
			{
			    fprintf(E->trace.fpt,"ERROR(compute elemental)-noway (3u5hd)\n");
			    fprintf(E->trace.fpt,"COMPEL: %f (%d)(%d)\n",E->trace.comp_el[j][kk],kk,E->trace.ieltrac[j][kk]);
			    fflush(E->trace.fpt);
			    exit(10);
			}
		}
	    if (iempty>0)
		{

		    /*
		      fprintf(E->trace.fpt,"%d empty elements filled with old values (%f percent)\n",iempty, (100.0*iempty/E->lmesh.nel));
		      fflush(E->trace.fpt);
		    */

		    if ((1.0*iempty/E->lmesh.nel)>0.80)
			{
			    fprintf(E->trace.fpt,"WARNING(compute_elemental...)-number of tracers is REALLY LOW\n");
			    fflush(E->trace.fpt);
			    if (E->trace.itracer_warnings==1) exit(10);
			}
		}

	    /* Fill oldel */


	    for (kk=1;kk<=E->lmesh.nel;kk++)
		{
		    E->trace.oldel[j][kk]=E->trace.comp_el[j][kk];
		}

	} /* end j */

    E->trace.istat_iempty=E->trace.istat_iempty+iempty;

    return;
}

/********** MAP COMPOSITION TO NODES ****************/
/*                                                  */


void map_composition_to_nodes(E)
     struct All_variables *E;
{

    int kk;
    int nelem, nodenum;
    int j;


    static int been_here=0;

    if (been_here==0)
	{

	    for (j=1;j<=E->sphere.caps_per_proc;j++)
		{

		    if ((E->trace.comp_node[j]=(double *)malloc((E->lmesh.nno+1)*sizeof(double)))==NULL)
			{
			    fprintf(E->trace.fpt,"AKM(map_compostion_to_nodes)-no memory 983rk\n");
			    fflush(E->trace.fpt);
			    exit(10);
			}
		}

	    been_here++;
	}

    for (j=1;j<=E->sphere.caps_per_proc;j++)
	{

	    /* first, initialize node array */

	    for (kk=1;kk<=E->lmesh.nno;kk++)
		{
		    E->trace.comp_node[j][kk]=0.0;
		}

	    /* Loop through all elements */

	    for (nelem=1;nelem<=E->lmesh.nel;nelem++)
		{

		    /* for each element, loop through element nodes */

		    /* weight composition */

		    for (nodenum=1;nodenum<=8;nodenum++)
			{

			    E->trace.comp_node[j][E->ien[j][nelem].node[nodenum]] +=
				E->trace.comp_el[j][nelem]*
				E->TWW[E->mesh.levmax][j][nelem].node[nodenum];

			}

		} /* end nelem */
	} /* end j */

    /* akm modified exchange node routine for doubles */

    (E->exchange_node_d)(E,E->trace.comp_node,E->mesh.levmax);

    /* Divide by nodal volume */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
	{
	    for (kk=1;kk<=E->lmesh.nno;kk++)
		{
		    E->trace.comp_node[j][kk] *= E->MASS[E->mesh.levmax][j][kk];
		}

	    /* testing */
	    /*
	      for (kk=1;kk<=E->lmesh.nel;kk++)
	      {
	      fprintf(E->trace.fpt,"%d %f\n",kk,E->trace.comp_el[j][kk]);
	      }

	      for (kk=1;kk<=E->lmesh.nno;kk++)
	      {
	      fprintf(E->trace.fpt,"%d %f %f\n",kk,E->sx[j][3][kk],E->trace.comp_node[j][kk]);
	      }
	    */


	} /* end j */

    return;
}


/*********** GET BULK COMPOSITION *******************************/

void get_bulk_composition(E)
     struct All_variables *E;

{

    char output_file[200];
    char input_s[1000];

    double return_bulk_value_d();
    double volume;
    double rdum1;
    double rdum2;
    double rdum3;

    int ival=0;
    int idum1;
    int istep;


    FILE *fp;

    static int been_here=0;


    /* ival=0 returns integral not average */

    volume=return_bulk_value_d(E,E->composition.comp_node,ival);

    E->composition.bulk_composition=volume;

    /* Here we assume if restart = 1 or 0 tracers are reset          */
    /*                if restart = 2 tracers may or may not be reset  */
    /*                   (read initial composition from file)         */


    if (been_here==0)
	{
	    if (E->composition.ireset_initial_composition==1)
		{
		    E->composition.initial_bulk_composition=volume;
		}
	    else
		{

		    if (E->trace.ic_method!=2)
			{
			    fprintf(E->trace.fpt,"ERROR(bulk composition)-wrong reset,restart combo\n");
			    fflush(E->trace.fpt);
			    exit(10);
			}

		    sprintf(output_file,"%s.comp.%d.%d",E->control.old_P_file,
			    E->parallel.me,E->monitor.solution_cycles);

		    fp=fopen(output_file,"r");
		    fgets(input_s,200,fp);
		    sscanf(input_s,"%d %d %lf %lf %lf",
			   &istep,&idum1,&rdum1,&rdum2,&rdum3);

		    E->composition.initial_bulk_composition=rdum2;
		    fclose(fp);

		    if (istep!=E->monitor.solution_cycles)
			{
			    fprintf(E->trace.fpt,"ERROR(get_bulk_composition) %d %d\n",
				    istep,E->monitor.solution_cycles);
			    fflush(E->trace.fpt);
			    exit(10);
			}
		}
	}

    E->composition.error_fraction=((volume-E->composition.initial_bulk_composition)/
			     E->composition.initial_bulk_composition);

    parallel_process_sync(E);

    been_here++;
    return;
}



/******************* READ COMP ***********************************************/
/*                                                                           */
/* This function is similar to read_temp. It is used to read the composition */
/* from file for post-proceesing.                                            */

void read_comp(E)
     struct All_variables *E;
{
    int i,ii,m,mm,ll;
    char output_file[255],input_s[1000];

    double g;
    FILE *fp;



    ii = E->monitor.solution_cycles;
    sprintf(output_file,"%s.comp.%d.%d",E->control.old_P_file,E->parallel.me,ii);

    if ((fp=fopen(output_file,"r"))==NULL)
        {
	    fprintf(stderr,"ERROR(read_temp) - %s not found\n",output_file);
	    fflush(stderr);
	    exit(10);
        }

    fgets(input_s,1000,fp);
    sscanf(input_s,"%d %d %f",&ll,&mm,&E->monitor.elapsed_time);

    for(m=1;m<=E->sphere.caps_per_proc;m++)
        {
	    E->trace.comp_node[m]=(double *)malloc((E->lmesh.nno+1)*sizeof(double));

	    fgets(input_s,1000,fp);
	    sscanf(input_s,"%d %d",&ll,&mm);
	    for(i=1;i<=E->lmesh.nno;i++)
		{
		    if (fgets(input_s,1000,fp)==NULL)
			{
			    fprintf(stderr,"ERROR(read_comp) -data for node %d not found\n",i);
			    fflush(stderr);
			    exit(10);
			}
		    sscanf(input_s,"%lf",&g);
		    E->trace.comp_node[m][i] = g;

		}
        }

    fclose (fp);
    return;
}


