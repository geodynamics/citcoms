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
/*

  Tracer_setup.c

      A program which initiates the distribution of tracers
      and advects those tracers in a time evolving velocity field.
      Called and used from the CitCOM finite element code.
      Written 2/96 M. Gurnis for Citcom in cartesian geometry
      Modified by Lijie in 1998 and by Vlad and Eh in 2005 for the
      regional version of CitcomS. In 2003, Allen McNamara wrote the
      tracer module for the global version of CitcomS. In 2007, Eh Tan
      merged the two versions of tracer codes together.
*/

#include <math.h>
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"

#ifdef USE_GGRD
#include "ggrd_handling.h"
#endif

#ifdef USE_GZDIR
int open_file_zipped(char *, FILE **,struct All_variables *);
void gzip_file(char *);
#endif

int icheck_that_processor_shell(struct All_variables *E,
                                       int j, int nprocessor, double rad);
void tracer_post_processing(struct All_variables *E);
void allocate_tracer_arrays(struct All_variables *E,
                            int j, int number_of_tracers);
void count_tracers_of_flavors(struct All_variables *E);

int full_icheck_cap(struct All_variables *E, int icap,
                    CartesianCoord cc, double rad);
int regional_icheck_cap(struct All_variables *E, int icap,
                        SphericalCoord sc, double rad);

static void find_tracers(struct All_variables *E);
static void predict_tracers(struct All_variables *E);
static void correct_tracers(struct All_variables *E);
static void make_tracer_array(struct All_variables *E);
static void generate_random_tracers(struct All_variables *E,
                                    int tracers_cap, int j);
static void read_tracer_file(struct All_variables *E);
static void read_old_tracer_file(struct All_variables *E);
static void check_sum(struct All_variables *E);
static int isum_tracers(struct All_variables *E);
static void init_tracer_flavors(struct All_variables *E);
int read_double_vector(FILE *, int , double *);
int icheck_processor_shell(struct All_variables *,
                           int , double );



void SphericalCoord::writeToMem(double *mem) const {
	mem[0] = _theta;
	mem[1] = _phi;
	mem[2] = _rad;
}

void SphericalCoord::readFromMem(const double *mem) {
	_theta = mem[0];
	_phi = mem[1];
	_rad = mem[2];
}

void CartesianCoord::writeToMem(double *mem) const {
	mem[0] = _x;
	mem[1] = _y;
	mem[2] = _z;
}

void CartesianCoord::readFromMem(const double *mem) {
	_x = mem[0];
	_y = mem[1];
	_z = mem[2];
}


size_t Tracer::size(void) {
	return 	_sc.size()	// spherical coords
	+ _cc.size()		// Cartesian coords
	+ _cc0.size()		// Original coords
	+ _Vc.size()		// Velocity
	+ 1					// flavor
	+ 1;				// ielement
}

void Tracer::writeToMem(double *mem) const {
	_sc.writeToMem(&mem[0]);
	_cc.writeToMem(&mem[3]);
	_cc0.writeToMem(&mem[6]);
	_Vc.writeToMem(&mem[9]);
	mem[12] = _flavor;
	mem[13] = _ielement;
}

void Tracer::readFromMem(const double *mem) {
	_sc.readFromMem(&mem[0]);
	_cc.readFromMem(&mem[3]);
	_cc0.readFromMem(&mem[6]);
	_Vc.readFromMem(&mem[9]);
	_flavor = mem[12];
	_ielement = mem[13];
}



SphericalCoord CartesianCoord::toSpherical(void) const
{
	double temp;
	double theta, phi, rad;
	
	temp=_x*_x+_y*_y;
	
	theta=atan2(sqrt(temp),_z);
	phi=myatan(_y,_x);
	rad=sqrt(temp+_z*_z);
	
	return SphericalCoord(theta, phi, rad);
}

CartesianCoord SphericalCoord::toCartesian(void) const
{
	double sint,cost,sinf,cosf;
	double x, y, z;
	
	sint=sin(_theta);
	cost=cos(_theta);
	sinf=sin(_phi);
	cosf=cos(_phi);
	
	x=_rad*sint*cosf;
	y=_rad*sint*sinf;
	z=_rad*cost;
	
	return CartesianCoord(x,y,z);
}



const CartesianCoord CartesianCoord::operator+(const CartesianCoord &other) const {
	return CartesianCoord(	this->_x+other._x,
						  this->_y+other._y,
						  this->_z+other._z);
}

// Multiply each element by a constant factor
const CartesianCoord CartesianCoord::operator*(const double &val) const {
	return CartesianCoord(	this->_x*val,
						  this->_y*val,
						  this->_z*val);
}

// Divide each element by a constant factor
const CartesianCoord CartesianCoord::operator/(const double &val) const {
	return CartesianCoord(	this->_x/val,
						  this->_y/val,
						  this->_z/val);
}


// Returns a constrained angle between 0 and 2 PI
double SphericalCoord::constrainAngle(const double angle) const {
    const double two_pi = 2.0*M_PI;
	
	return angle - two_pi * floor(angle/two_pi);
}

// Constrains theta to be between 0 and PI while simultaneously constraining phi between 0 and 2 PI
void SphericalCoord::constrainThetaPhi(void) {
	_theta = constrainAngle(_theta);
	if (_theta > M_PI) {
		_theta = 2*M_PI - _theta;
		_phi += M_PI;
	}
	_phi = constrainAngle(_phi);
}


void CapBoundary::setBoundary(int bnum, CartesianCoord cc, SphericalCoord sc) {
	assert(bnum>=0 && bnum < 4);
	cartesian_boundary[bnum] = cc;
	spherical_boundary[bnum] = sc;
	cos_theta[bnum] = cos(sc._theta);
	sin_theta[bnum] = sin(sc._theta);
	cos_phi[bnum] = cos(sc._phi);
	sin_phi[bnum] = sin(sc._phi);
}


void tracer_input(struct All_variables *E)
{
    void full_tracer_input();
    void myerror();
    void report();
    char message[100];
    int m=E->parallel.me;
    int i;

    input_boolean("tracer",&(E->control.tracer),"off",m);
    input_boolean("tracer_enriched",
		  &(E->control.tracer_enriched),"off",m);
    if(E->control.tracer_enriched){
      if(!E->control.tracer)	/* check here so that we can get away
				   with only one if statement in
				   Advection_diffusion */
	myerror(E,"need to switch on tracers for tracer_enriched");

      input_float("Q0_enriched",&(E->control.Q0ER),"0.0",m);
      snprintf(message,100,"using compositionally enriched heating: C = 0: %g C = 1: %g (only one composition!)",
	       E->control.Q0,E->control.Q0ER);
      report(E,message);
      //
      // this check doesn't work at this point in the code, and we didn't want to put it into every call to
      // Advection diffusion
      //
      //if(E->composition.ncomp != 1)
      //myerror(E,"enriched tracers cannot deal with more than one composition yet");

    }
    if(E->control.tracer) {

        /* tracer_ic_method=0 (random generated array) */
        /* tracer_ic_method=1 (all proc read the same file) */
        /* tracer_ic_method=2 (each proc reads its restart file) */
        input_int("tracer_ic_method",&(E->trace.ic_method),"0,0,nomax",m);

        if (E->trace.ic_method==0){
            input_int("tracers_per_element",&(E->trace.itperel),"10,0,nomax",m);
	}
        else if (E->trace.ic_method==1)
            input_string("tracer_file",E->trace.tracer_file,"tracer.dat",m);
        else if (E->trace.ic_method==2) {
            /* Use 'datadir_old', 'datafile_old', and 'solution_cycles_init' */
            /* to form the filename */
        }
        else {
            fprintf(stderr,"Sorry, tracer_ic_method only 0, 1 and 2 available\n");
            parallel_process_termination();
        }


        /* How many flavors of tracers */
        /* If tracer_flavors > 0, each element will report the number of
         * tracers of each flavor inside it. This information can be used
         * later for many purposes. One of it is to compute composition,
         * either using absolute method or ratio method. */
        input_int("tracer_flavors",&(E->trace.nflavors),"0,0,nomax",m);

	/* 0: default from layers 
	   1: from netcdf grds
	   
	   
	   99: from grds, overriding checkpoints during restart
	   (1 and 99 require ggrd)
	*/

        input_int("ic_method_for_flavors",
		  &(E->trace.ic_method_for_flavors),"0,0,nomax",m);


        if (E->trace.nflavors > 1) {
            switch(E->trace.ic_method_for_flavors){
	      /* default method */
            case 0:			
	      /* flavors initialized from layers */
                E->trace.z_interface = (double*) malloc((E->trace.nflavors-1)
                                                        *sizeof(double));
                for(i=0; i<E->trace.nflavors-1; i++)
                    E->trace.z_interface[i] = 0.7;

                input_double_vector("z_interface", E->trace.nflavors-1,
                                    E->trace.z_interface, m);
                break;
		/* 
		   two grd init method, second will override restart
		*/
#ifdef USE_GGRD
            case 1:
	    case 99:		/* will override restart */
	      /* from grid in top n materials, this will override
		 the checkpoint input */
	      input_string("ictracer_grd_file",E->trace.ggrd_file,"",m); /* file from which to read */
	      input_int("ictracer_grd_layers",&(E->trace.ggrd_layers),"2",m); /* 

									      >0 : which top layers to use, layer <= ictracer_grd_layers
									      <0 : only use one layer layer == -ictracer_grd_layers

									      */
	      break;
	      
#endif
            default:
                fprintf(stderr,"ic_method_for_flavors %i undefined (1 and 99 only for ggrd mode)\n",E->trace.ic_method_for_flavors);
                parallel_process_termination();
                break;
            }
        }

        /* Warning level */
        input_boolean("itracer_warnings",&(E->trace.itracer_warnings),"on",m);


        if(E->parallel.nprocxy == 12)
            full_tracer_input(E);


        composition_input(E);

    }

    return;
}


void tracer_initial_settings(struct All_variables *E)
{
   E->trace.advection_time = 0;
   E->trace.find_tracers_time = 0;
   E->trace.lost_souls_time = 0;

   if(E->parallel.nprocxy == 1) {
       E->problem_tracer_setup = regional_tracer_setup;

       E->trace.keep_within_bounds = regional_keep_within_bounds;
       E->trace.get_velocity = regional_get_velocity;
       E->trace.iget_element = regional_iget_element;
   }
   else {
       E->problem_tracer_setup = full_tracer_setup;

       E->trace.keep_within_bounds = full_keep_within_bounds;
       E->trace.get_velocity = full_get_velocity;
       E->trace.iget_element = full_iget_element;
   }
}



/*****************************************************************************/
/* This function is the primary tracing routine called from Citcom.c         */
/* In this code, unlike the original 3D cartesian code, force is filled      */
/* during Stokes solution. No need to call thermal_buoyancy() after tracing. */


void tracer_advection(struct All_variables *E)
{
    double CPU_time0();
    double begin_time = CPU_time0();

    /* advect tracers */
    predict_tracers(E);
    correct_tracers(E);

    /* check that the number of tracers is conserved */
    check_sum(E);

    /* count # of tracers of each flavor */
    if (E->trace.nflavors > 0)
        count_tracers_of_flavors(E);

    /* update the composition field */
    if (E->composition.on) {
        fill_composition(E);
    }

    E->trace.advection_time += CPU_time0() - begin_time;

    tracer_post_processing(E);
}



/********* TRACER POST PROCESSING ****************************************/

void tracer_post_processing(struct All_variables *E)
{
    int i;

    /* reset statistical counters */
    E->trace.istat_isend=0;
    E->trace.istat_elements_checked=0;
    E->trace.istat1=0;

    /* write timing information every 20 steps */
    if ((E->monitor.solution_cycles % 20) == 0) {
        fprintf(E->trace.fpt, "STEP %d\n", E->monitor.solution_cycles);

        fprintf(E->trace.fpt, "Advecting tracers takes %f seconds.\n",
                E->trace.advection_time - E->trace.find_tracers_time);
        fprintf(E->trace.fpt, "Finding element takes %f seconds.\n",
                E->trace.find_tracers_time - E->trace.lost_souls_time);
        fprintf(E->trace.fpt, "Exchanging lost tracers takes %f seconds.\n",
                E->trace.lost_souls_time);
    }

    if(E->control.verbose){
      fprintf(E->trace.fpt,"Number of times for all element search  %d\n",E->trace.istat1);

      fprintf(E->trace.fpt,"Number of tracers sent to other processors: %d\n",E->trace.istat_isend);

      fprintf(E->trace.fpt,"Number of times element columns are checked: %d \n",E->trace.istat_elements_checked);

      /* compositional and error fraction data files */
      //TODO: move
      if (E->composition.on) {
        fprintf(E->trace.fpt,"Empty elements filled with old compositional "
                "values: %d (%f percent)\n", E->trace.istat_iempty,
                (100.0*E->trace.istat_iempty)/E->lmesh.nel);
        E->trace.istat_iempty=0;


        get_bulk_composition(E);

        if (E->parallel.me==0) {

            fprintf(E->fp,"composition: %e",E->monitor.elapsed_time);
            for (i=0; i<E->composition.ncomp; i++)
                fprintf(E->fp," %e", E->composition.bulk_composition[i]);
            fprintf(E->fp,"\n");

            fprintf(E->fp,"composition_error_fraction: %e",E->monitor.elapsed_time);
            for (i=0; i<E->composition.ncomp; i++)
                fprintf(E->fp," %e", E->composition.error_fraction[i]);
            fprintf(E->fp,"\n");

        }
      }
      fflush(E->trace.fpt);
    }

    return;
}


/*********** PREDICT TRACERS **********************************************/
/*                                                                        */
/* This function predicts tracers performing an euler step                */
/*                                                                        */

static void predict_tracers(struct All_variables *E)
{
    int						j, nelem;
    double					dt;
	SphericalCoord			sc0, sc_pred;
	CartesianCoord			cc0, cc_pred, velocity_vector;
	TracerList::iterator	tr;
	
    dt=E->advection.timestep;
	
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
		
        for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();++tr) {
			// Get initial tracer position
			sc0 = tr->getSphericalPos();
			cc0 = tr->getCartesianPos();
			
			// Interpolate the velocity of the tracer from element it is in
            nelem = tr->ielement();
            velocity_vector = (E->trace.get_velocity)(E,j,nelem,sc0);
			
			// Advance the tracer location in an Euler step
			cc_pred = cc0 + velocity_vector * dt;
			
            // Ensure tracer remains in the boundaries
			sc_pred = cc_pred.toSpherical();
            (E->trace.keep_within_bounds)(E, cc_pred, sc_pred);
			
			// Set new tracer coordinates
            tr->setCoords(sc_pred, cc_pred);
			
            // Save original coords and velocities for later correction
			
			tr->setOrigVals(cc0, velocity_vector);
			
        } /* end predicting tracers */
    } /* end caps */
	
    /* find new tracer elements and caps */
    find_tracers(E);
}

/*********** CORRECT TRACERS **********************************************/
/*                                                                        */
/* This function corrects tracers using both initial and                  */
/* predicted velocities                                                   */
/*                                                                        */

static void correct_tracers(struct All_variables *E)
{
    int						j, nelem;
    double					dt;
	TracerList::iterator	tr;
	CartesianCoord			orig_pos, orig_vel, pred_vel, new_coord;
	SphericalCoord			new_coord_sph;
	
    dt=E->advection.timestep;
	
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
        for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();++tr) {
			
			// Get the original tracer position and velocity
            orig_pos = tr->getOrigCartesianPos();
            orig_vel = tr->getCartesianVel();
			
			// Get the predicted velocity by interpolating in the current element
            nelem = tr->ielement();
            pred_vel = (E->trace.get_velocity)(E, j, nelem, tr->getSphericalPos());
			
			// Correct the tracer position based on the original and new velocities
			new_coord = orig_pos + (orig_vel+pred_vel) * (dt*0.5);
			new_coord_sph = new_coord.toSpherical();
			
			// Ensure tracer remains in boundaries
            (E->trace.keep_within_bounds)(E, new_coord, new_coord_sph);
			
            // Fill in current positions (original values are no longer important)
			tr->setCoords(new_coord_sph, new_coord);
			
        } /* end correcting tracers */
    } /* end caps */
	
    /* find new tracer elements and caps */
    find_tracers(E);
}


/************ FIND TRACERS *************************************/
/*                                                             */
/* This function finds tracer elements and moves tracers to    */
/* other processor domains if necessary.                       */
/* Array ielement is filled with elemental values.                */

static void find_tracers(struct All_variables *E)
{

    int						iel, j, iprevious_element;
	TracerList::iterator	tr;
    double					begin_time = CPU_time0();

    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        /* initialize arrays and statistical counters */

        E->trace.istat1=0;
        for (int kk=0;kk<=4;kk++) {
            E->trace.istat_ichoice[j][kk]=0;
        }

        for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();) {

			CartesianCoord		cc;
			SphericalCoord		sc;
			
			sc = tr->getSphericalPos();
			cc = tr->getCartesianPos();

            iprevious_element=tr->ielement();

            iel=(E->trace.iget_element)(E,j,iprevious_element,cc,sc);
            ///* debug *
            fprintf(E->trace.fpt,"BB. kk %d %d %d %f %f %f %f %f %f\n",
					j,iprevious_element,iel,cc._x,cc._y,cc._z,sc._theta,sc._phi,sc._rad);
            fflush(E->trace.fpt);
            /**/

            tr->set_ielement(iel);

            if (iel == -99) {
                /* tracer is inside other processors */
				E->trace.escaped_tracers[j].push_back(*tr);
				tr=E->trace.tracers[j].erase(tr);
				//fprintf(stderr, "ejected!\n" );
            } else if (iel == -1) {
                /* tracer is inside this processor,
                 * but cannot find its element.
                 * Throw away the tracer. */

                if (E->trace.itracer_warnings) exit(10);

				tr=E->trace.tracers[j].erase(tr);
            } else {
				tr++;
			}

        } /* end tracers */

    } /* end j */


    /* Now take care of tracers that exited cap */

    if (E->parallel.nprocxy == 12)
        full_lost_souls(E);
    else
        regional_lost_souls(E);


    E->trace.find_tracers_time += CPU_time0() - begin_time;

    return;
}


/***********************************************************************/
/* This function computes the number of tracers in each element.       */
/* Each tracer can be of different "flavors", which is the 0th index   */
/* of extraq. How to interprete "flavor" is left for the application.  */

void count_tracers_of_flavors(struct All_variables *E)
{

    int j, flavor, e, kk;
	TracerList::iterator	tr;

    for (j=1; j<=E->sphere.caps_per_proc; j++) {

        /* first zero arrays */
        for (flavor=0; flavor<E->trace.nflavors; flavor++)
            for (e=1; e<=E->lmesh.nel; e++)
                E->trace.ntracer_flavor[j][flavor][e] = 0;

        /* Fill arrays */
        for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();++tr) {
            e = tr->ielement();
            flavor = tr->flavor();
            E->trace.ntracer_flavor[j][flavor][e]++;
        }
    }

    /* debug */
    /**
    for (j=1; j<=E->sphere.caps_per_proc; j++) {
        for (e=1; e<=E->lmesh.nel; e++) {
            fprintf(E->trace.fpt, "element=%d ntracer_flaver =", e);
            for (flavor=0; flavor<E->trace.nflavors; flavor++) {
                fprintf(E->trace.fpt, " %d",
                        E->trace.ntracer_flavor[j][flavor][e]);
            }
            fprintf(E->trace.fpt, "\n");
        }
    }
    fflush(E->trace.fpt);
    /**/

    return;
}



void initialize_tracers(struct All_variables *E)
{

	E->trace.tracers = new TracerList[13];
	E->trace.escaped_tracers = new TracerList[13];
	
    if (E->trace.ic_method==0)
        make_tracer_array(E);
    else if (E->trace.ic_method==1)
        read_tracer_file(E);
    else if (E->trace.ic_method==2)
        read_old_tracer_file(E);
    else {
        fprintf(E->trace.fpt,"Not ready for other inputs yet\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }

    /* total number of tracers  */

    E->trace.ilast_tracer_count = isum_tracers(E);
    fprintf(E->trace.fpt, "Sum of Tracers: %d\n", E->trace.ilast_tracer_count);
    if(E->parallel.me==0)
        fprintf(stderr, "Sum of Tracers: %d\n", E->trace.ilast_tracer_count);

    /* find elements */

    find_tracers(E);

    /* count # of tracers of each flavor */

    if (E->trace.nflavors > 0)
        count_tracers_of_flavors(E);
	
    return;
}


/************** MAKE TRACER ARRAY ********************************/
/* Here, each processor will generate tracers somewhere          */
/* in the sphere - check if its in this cap  - then check radial */

static void make_tracer_array(struct All_variables *E)
{

    int tracers_cap;
    int j;
    double processor_fraction;

    if (E->parallel.me==0) fprintf(stderr,"Making Tracer Array\n");

    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        processor_fraction=E->lmesh.volume/E->mesh.volume;
        tracers_cap=E->mesh.nel*E->trace.itperel*processor_fraction;
        /*
          fprintf(stderr,"AA: proc frac: %f (%d) %d %d %f %f\n",processor_fraction,tracers_cap,E->lmesh.nel,E->parallel.nprocz, E->sx[j][3][E->lmesh.noz],E->sx[j][3][1]);
        */

        fprintf(E->trace.fpt,"\nGenerating %d Tracers\n",tracers_cap);

        generate_random_tracers(E, tracers_cap, j);

    }/* end j */


    /* Initialize tracer flavors */
    if (E->trace.nflavors) init_tracer_flavors(E);
}



static void generate_random_tracers(struct All_variables *E,
                                    int tracers_cap, int j)
{
    int kk;
    int ival;
    int number_of_tries=0;
    int max_tries;

    double xmin,xmax,ymin,ymax,zmin,zmax;
    double random1,random2,random3;

    allocate_tracer_arrays(E,j,tracers_cap);

    /* Finding the min/max of the cartesian coordinates. */
    /* One must loop over E->X to find the min/max, since the 8 corner */
    /* nodes may not be the min/max. */
    xmin = ymin = zmin = E->sphere.ro;
    xmax = ymax = zmax = -E->sphere.ro;
    for (kk=1; kk<=E->lmesh.nno; kk++) {
		double x,y,z;
        x = E->x[j][1][kk];
        y = E->x[j][2][kk];
        z = E->x[j][3][kk];

        xmin = ((xmin < x) ? xmin : x);
        xmax = ((xmax > x) ? xmax : x);
        ymin = ((ymin < y) ? ymin : y);
        ymax = ((ymax > y) ? ymax : y);
        zmin = ((zmin < z) ? zmin : z);
        zmax = ((zmax > z) ? zmax : z);
    }

    /* Tracers are placed randomly in cap */
    /* (intentionally using rand() instead of srand() )*/
    while (E->trace.tracers[j].size()<tracers_cap) {

        number_of_tries++;
        max_tries=100*tracers_cap;

        if (number_of_tries>max_tries) {
            fprintf(E->trace.fpt,"Error(make_tracer_array)-too many tries?\n");
            fprintf(E->trace.fpt,"%d %d %d\n",max_tries,number_of_tries,RAND_MAX);
            fflush(E->trace.fpt);
            exit(10);
        }

#if 1
        random1=drand48();
        random2=drand48();
        random3=drand48();
#else  /* never called */
        random1=(1.0*rand())/(1.0*RAND_MAX);
        random2=(1.0*rand())/(1.0*RAND_MAX);
        random3=(1.0*rand())/(1.0*RAND_MAX);
#endif

		CartesianCoord	new_cc;
		SphericalCoord	new_sc;
		
        new_cc = CartesianCoord(xmin+random1*(xmax-xmin),
								ymin+random2*(ymax-ymin),
								zmin+random3*(zmax-zmin));

        /* first check if within shell */

		new_sc = new_cc.toSpherical();

        if (new_sc._rad>=E->sx[j][3][E->lmesh.noz]) continue;
        if (new_sc._rad<E->sx[j][3][1]) continue;

        /* check if in current cap */
        if (E->parallel.nprocxy==1)
            ival=regional_icheck_cap(E,0,new_sc,new_sc._rad);
        else
            ival=full_icheck_cap(E,0,new_cc,new_sc._rad);

        if (ival!=1) continue;

        /* Made it, so record tracer information */

        (E->trace.keep_within_bounds)(E, new_cc, new_sc);

		Tracer	new_tracer;
		
		new_tracer.setCoords(new_sc, new_cc);
		E->trace.tracers[j].push_back(new_tracer);
    } /* end while */

    return;
}


/******** READ TRACER ARRAY *********************************************/
/*                                                                      */
/* This function reads tracers from input file.                         */
/* All processors read the same input file, then sort out which ones    */
/* belong.                                                              */

static void read_tracer_file(struct All_variables *E)
{

    char input_s[1000];

    int number_of_tracers, ncolumns;
    int kk;
    int icheck;
    int iestimate;
    int icushion;
    int i, j;

    double buffer[100];

    FILE *fptracer;

    fptracer=fopen(E->trace.tracer_file,"r");

    fgets(input_s,200,fptracer);
    if(sscanf(input_s,"%d %d",&number_of_tracers,&ncolumns) != 2) {
        fprintf(stderr,"Error while reading file '%s'\n", E->trace.tracer_file);
        exit(8);
    }
    fprintf(E->trace.fpt,"%d Tracers, %d columns in file \n",
            number_of_tracers, ncolumns);

    /* some error control */
    if (1+3 != ncolumns) {
        fprintf(E->trace.fpt,"ERROR(read tracer file)-wrong # of columns\n");
        fflush(E->trace.fpt);
        exit(10);
    }

    /* initially size tracer arrays to number of tracers divided by processors */

    icushion=100;

    /* for absolute tracer method */
    E->trace.number_of_tracers = number_of_tracers;

    iestimate=number_of_tracers/E->parallel.nproc + icushion;

    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        allocate_tracer_arrays(E,j,iestimate);

        for (kk=1;kk<=number_of_tracers;kk++) {
			SphericalCoord		in_coord_sph;
			CartesianCoord		in_coord_cc;
            int					len, ncol;
			
            ncol = 3 + 1;	// ERIC: E->trace.number_of_extra_quantities;

            len = read_double_vector(fptracer, ncol, buffer);
            if (len != ncol) {
                fprintf(E->trace.fpt,"ERROR(read tracer file) - wrong input file format: %s\n", E->trace.tracer_file);
                fflush(E->trace.fpt);
                exit(10);
            }

			in_coord_sph.readFromMem(buffer);
			in_coord_cc = in_coord_sph.toCartesian();

            /* make sure theta, phi is in range, and radius is within bounds */

            (E->trace.keep_within_bounds)(E,in_coord_cc,in_coord_sph);

            /* check whether tracer is within processor domain */

            icheck=1;
            if (E->parallel.nprocz>1) icheck=icheck_processor_shell(E,j,in_coord_sph._rad);
            if (icheck!=1) continue;

            if (E->parallel.nprocxy==1)
                icheck=regional_icheck_cap(E,0,in_coord_sph,in_coord_sph._rad);
            else
                icheck=full_icheck_cap(E,0,in_coord_cc,in_coord_sph._rad);

            if (icheck==0) continue;

            /* if still here, tracer is in processor domain */

			Tracer new_tracer;
			
			new_tracer.setCoords(in_coord_sph, in_coord_cc);
			new_tracer.set_flavor(buffer[3]);
			E->trace.tracers[j].push_back(new_tracer);

        } /* end kk, number of tracers */

        fprintf(E->trace.fpt,"Number of tracers in this cap is: %d\n",
                E->trace.tracers[j].size());

        /** debug **
        for (kk=1; kk<=E->trace.ntracers[j]; kk++) {
            fprintf(E->trace.fpt, "tracer#=%d sph_coord=(%g,%g,%g)", kk,
                    E->trace.basicq[j][0][kk],
                    E->trace.basicq[j][1][kk],
                    E->trace.basicq[j][2][kk]);
            fprintf(E->trace.fpt, "   extraq=");
            for (i=0; i<E->trace.number_of_extra_quantities; i++)
                fprintf(E->trace.fpt, " %g", E->trace.extraq[j][i][kk]);
            fprintf(E->trace.fpt, "\n");
        }
        fflush(E->trace.fpt);
        /**/

    } /* end j */

    fclose(fptracer);

    icheck=isum_tracers(E);

    if (icheck!=number_of_tracers) {
        fprintf(E->trace.fpt,"ERROR(read_tracer_file) - tracers != number in file\n");
        fprintf(E->trace.fpt,"Tracers in system: %d\n", icheck);
        fprintf(E->trace.fpt,"Tracers in file: %d\n", number_of_tracers);
        fflush(E->trace.fpt);
        exit(10);
    }

    return;
}


/************** READ OLD TRACER FILE *************************************/
/*                                                                       */
/* This function read tracers written from previous calculation          */
/* and the tracers are read as seperate files for each processor domain. */

static void read_old_tracer_file(struct All_variables *E)
{

    char output_file[200];
    char input_s[1000];

    int i,j,kk,rezip;
    int idum1,ncolumns;
    int numtracers;

    double rdum1;
    double theta,phi,rad;
    double x,y,z;
    double buffer[100];

    FILE *fp1;

    /* deal with different output formats */
#ifdef USE_GZDIR
    if(strcmp(E->output.format, "ascii-gz") == 0){
      sprintf(output_file,"%s/%d/tracer.%d.%d",
	      E->control.data_dir_old,E->monitor.solution_cycles_init,E->parallel.me,E->monitor.solution_cycles_init);
      rezip = open_file_zipped(output_file,&fp1,E);
    }else{
      sprintf(output_file,"%s.tracer.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
      if ( (fp1=fopen(output_file,"r"))==NULL) {
        fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-gziped file not found %s\n",output_file);
        fflush(E->trace.fpt);
        exit(10);
      }
    }
#else
    sprintf(output_file,"%s.tracer.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
    if ( (fp1=fopen(output_file,"r"))==NULL) {
        fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-file not found %s\n",output_file);
        fflush(E->trace.fpt);
        exit(10);
    }
#endif

    fprintf(stderr,"Read old tracers from %s\n",output_file);


    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        fgets(input_s,200,fp1);
        if(sscanf(input_s,"%d %d %d %lf",
                  &idum1, &numtracers, &ncolumns, &rdum1) != 4) {
            fprintf(stderr,"Error while reading file '%s'\n", output_file);
            exit(8);
        }


        /* some error control */
        // ERIC: if (E->trace.number_of_extra_quantities+3 != ncolumns) {
        if (1+3 != ncolumns) {
            fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-wrong # of columns\n");
            fflush(E->trace.fpt);
            exit(10);
        }

        /* allocate memory for tracer arrays */

        allocate_tracer_arrays(E,j,numtracers);

        for (kk=1;kk<=numtracers;kk++) {
            int len, ncol;
			SphericalCoord	sc;
			CartesianCoord	cc;
			
            ncol = 3 + 1;

            len = read_double_vector(fp1, ncol, buffer);
            if (len != ncol) {
                fprintf(E->trace.fpt,"ERROR(read_old_tracer_file) - wrong input file format: %s\n", output_file);
                fflush(E->trace.fpt);
                exit(10);
            }

			sc.readFromMem(buffer);
			cc = sc.toCartesian();

            /* it is possible that if on phi=0 boundary, significant digits can push phi over 2pi */

            (E->trace.keep_within_bounds)(E,cc,sc);

			Tracer	new_tracer;
			
            new_tracer.setCoords(sc, cc);
			new_tracer.set_flavor(buffer[3]);
			E->trace.tracers[j].push_back(new_tracer);

        }

        /** debug **
        for (kk=1; kk<=E->trace.ntracers[j]; kk++) {
            fprintf(E->trace.fpt, "tracer#=%d sph_coord=(%g,%g,%g)", kk,
                    E->trace.basicq[j][0][kk],
                    E->trace.basicq[j][1][kk],
                    E->trace.basicq[j][2][kk]);
            fprintf(E->trace.fpt, "   extraq=");
            for (i=0; i<E->trace.number_of_extra_quantities; i++)
                fprintf(E->trace.fpt, " %g", E->trace.extraq[j][i][kk]);
            fprintf(E->trace.fpt, "\n");
        }
        fflush(E->trace.fpt);
        /**/

        fprintf(E->trace.fpt,"Read %d tracers from file %s\n",numtracers,output_file);
        fflush(E->trace.fpt);

    }
    fclose(fp1);
#ifdef USE_GZDIR
    if(strcmp(E->output.format, "ascii-gz") == 0)
      if(rezip)			/* rezip */
	gzip_file(output_file);
#endif

    return;
}


/*********** CHECK SUM **************************************************/
/*                                                                      */
/* This functions checks to make sure number of tracers is preserved    */

static void check_sum(struct All_variables *E)
{

    int number, iold_number;

    number = isum_tracers(E);

    iold_number = E->trace.ilast_tracer_count;

    if (number != iold_number) {
        fprintf(E->trace.fpt,"ERROR(check_sum)-break in conservation %d %d\n",
                number,iold_number);
        fflush(E->trace.fpt);
        if (E->trace.itracer_warnings)
            parallel_process_termination();
    }

    E->trace.ilast_tracer_count = number;

    return;
}


/************* ISUM TRACERS **********************************************/
/*                                                                       */
/* This function uses MPI to sum all tracers and returns number of them. */

static int isum_tracers(struct All_variables *E)
{
    int imycount;
    int iallcount;
    int j;

    iallcount = 0;

    imycount = 0;
    for (j=1; j<=E->sphere.caps_per_proc; j++)
        imycount = imycount + E->trace.tracers[j].size();

    MPI_Allreduce(&imycount,&iallcount,1,MPI_INT,MPI_SUM,E->parallel.world);

    return iallcount;
}



static void init_tracer_flavors(struct All_variables *E)
{
    int						j, kk, i;
    double					flavor, rad;
	TracerList::iterator	tr;
	
    switch(E->trace.ic_method_for_flavors){
		case 0:
			/* ic_method_for_flavors == 0 (layered structure) */
			/* any tracer above z_interface[i] is of flavor i */
			/* any tracer below z_interface is of flavor (nflavors-1) */
			for (j=1;j<=E->sphere.caps_per_proc;j++) {
				
				for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();++tr) {
					rad = tr->rad();
					
					flavor = E->trace.nflavors - 1;
					for (i=0; i<E->trace.nflavors-1; i++) {
						if (rad > E->trace.z_interface[i]) {
							flavor = i;
							break;
						}
					}
					tr->set_flavor(flavor);
				}
			}
			break;
			
		case 1:			/* from grd in top n layers */
		case 99:			/* (will override restart) */
#ifndef USE_GGRD
			fprintf(stderr,"ic_method_for_flavors %i requires the ggrd routines from hc, -DUSE_GGRD\n",
					E->trace.ic_method_for_flavors);
			parallel_process_termination();
#else
			ggrd_init_tracer_flavors(E);
#endif
			break;
			
			
		default:
			
			fprintf(stderr,"ic_method_for_flavors %i undefined\n",E->trace.ic_method_for_flavors);
			parallel_process_termination();
			break;
    }
	
    return;
}


/******************* get_neighboring_caps ************************************/
/*                                                                           */
/* Communicate with neighboring processors to get their cap boundaries,      */
/* which is later used by (E->trace.icheck_cap)()                            */
/*                                                                           */

void get_neighboring_caps(struct All_variables *E)
{
    const int ncorners = 4; /* # of top corner nodes */
    int i, j, n, d, kk, lev, idb;
    int num_ngb, neighbor_proc, tag;
    MPI_Status status[200];
    MPI_Request request[200];

    int node[ncorners];
    double xx[ncorners*2], rr[12][ncorners*2];
    int nox,noy,noz;
    double x,y,z;
    double theta,phi,rad;

    nox=E->lmesh.nox;
    noy=E->lmesh.noy;
    noz=E->lmesh.noz;

    node[0]=nox*noz*(noy-1)+noz;
    node[1]=noz;
    node[2]=noz*nox;
    node[3]=noz*nox*noy;

    lev = E->mesh.levmax;
    tag = 45;

    for (j=1; j<=E->sphere.caps_per_proc; j++) {

        /* loop over top corners to get their coordinates */
        n = 0;
        for (i=0; i<ncorners; i++) {
            for (d=0; d<2; d++) {
                xx[n] = E->sx[j][d+1][node[i]];
                n++;
            }
        }

        idb = 0;
        num_ngb = E->parallel.TNUM_PASS[lev][j];
        for (kk=1; kk<=num_ngb; kk++) {
            neighbor_proc = E->parallel.PROCESSOR[lev][j].pass[kk];

            MPI_Isend(xx, n, MPI_DOUBLE, neighbor_proc,
                      tag, E->parallel.world, &request[idb]);
            idb++;

            MPI_Irecv(rr[kk], n, MPI_DOUBLE, neighbor_proc,
                      tag, E->parallel.world, &request[idb]);
            idb++;
        }

        /* Storing the current cap information */
        for (i=0; i<n; i++)
            rr[0][i] = xx[i];

        /* Wait for non-blocking calls to complete */

        MPI_Waitall(idb, request, status);

        /* Storing the received cap information
         * XXX: this part assumes:
         *      1) E->sphere.caps_per_proc==1
         *      2) E->mesh.nsd==3
         */
        for (kk=0; kk<=num_ngb; kk++) {
            n = 0;
            for (i=0; i<ncorners; i++) {
				SphericalCoord	sc;
				CartesianCoord	cc;
				
                theta = rr[kk][n++];
                phi = rr[kk][n++];
                rad = E->sphere.ro;
				
				sc = SphericalCoord(theta, phi, rad);
				cc = sc.toCartesian();
				
				E->trace.boundaries[kk].setBoundary(i, cc, sc);
            }
        } /* end kk, number of neighbors */

        /* debugging output *
        for (kk=0; kk<=num_ngb; kk++) {
            if (kk==0)
                neighbor_proc = E->parallel.me;
            else
                neighbor_proc = E->parallel.PROCESSOR[lev][1].pass[kk];

            for (i=1; i<=ncorners; i++) {
                fprintf(E->trace.fpt, "pass=%d rank=%d corner=%d "
                        "sx=(%g, %g, %g)\n",
                        kk, neighbor_proc, i,
                        E->trace.theta_cap[kk][i],
                        E->trace.phi_cap[kk][i],
                        E->trace.rad_cap[kk][i]);
            }
        }
        fflush(E->trace.fpt);
        /**/
    }

    return;
}


/**************** INITIALIZE TRACER ARRAYS ************************************/
/*                                                                            */
/* This function allocates memories to tracer arrays.                         */

void allocate_tracer_arrays(struct All_variables *E,
                            int j, int number_of_tracers)
{

    int kk;

    if (E->trace.nflavors > 0) {
        E->trace.ntracer_flavor[j]=(int **)malloc(E->trace.nflavors*sizeof(int*));
        for (kk=0;kk<E->trace.nflavors;kk++) {
            if ((E->trace.ntracer_flavor[j][kk]=(int *)malloc((E->lmesh.nel+1)*sizeof(int)))==NULL) {
                fprintf(E->trace.fpt,"ERROR(initialize tracer arrays)-no memory 1c.%d\n",kk);
                fflush(E->trace.fpt);
                exit(10);
            }
        }
    }

    fflush(E->trace.fpt);

    return;
}


/********** ICHECK PROCESSOR SHELL *************/
/* returns -99 if rad is below current shell  */
/* returns 0 if rad is above current shell    */
/* returns 1 if rad is within current shell   */
/*                                            */
/* Shell, here, refers to processor shell     */
/*                                            */
/* shell is defined as bottom boundary up to  */
/* and not including the top boundary unless  */
/* the shell in question is the top shell     */

int icheck_processor_shell(struct All_variables *E,
                           int j, double rad)
{

    const int noz = E->lmesh.noz;
    const int nprocz = E->parallel.nprocz;
    double top_r, bottom_r;

    if (nprocz==1) return 1;

    top_r = E->sx[j][3][noz];
    bottom_r = E->sx[j][3][1];

    /* First check bottom */

    if (rad<bottom_r) return -99;

    /* Check top */

    if (rad<top_r) return 1;

    /* top processor */

    if ( (rad<=top_r) && (E->parallel.me_loc[3]==nprocz-1) ) return 1;

    /* If here, means point is above processor */
    return 0;
}


/********* ICHECK THAT PROCESSOR SHELL ********/
/*                                            */
/* Checks whether a given radius is within    */
/* a given processors radial domain.          */
/* Returns 0 if not, 1 if so.                 */
/* The domain is defined as including the bottom */
/* radius, but excluding the top radius unless   */
/* we the processor domain is the one that       */
/* is at the surface (then both boundaries are   */
/* included).                                    */

int icheck_that_processor_shell(struct All_variables *E,
                                int j, int nprocessor, double rad)
{
    int icheck_processor_shell();
    int me = E->parallel.me;

    /* nprocessor is right on top of me */
    if (nprocessor == me+1) {
        if (icheck_processor_shell(E, j, rad) == 0) return 1;
        else return 0;
    }

    /* nprocessor is right on bottom of me */
    if (nprocessor == me-1) {
        if (icheck_processor_shell(E, j, rad) == -99) return 1;
        else return 0;
    }

    /* Shouldn't be here */
    fprintf(E->trace.fpt, "Should not be here\n");
    fprintf(E->trace.fpt, "Error(check_shell) nprocessor: %d, radius: %f\n",
            nprocessor, rad);
    fflush(E->trace.fpt);
    exit(10);

    return 0;
}


