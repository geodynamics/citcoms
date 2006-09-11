/*
 * CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
 * Copyright (C) 2002-2005, California Institute of Technology.
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include "hdf5.h"


#define true 1
#define false 0
#define STRING_BUFFER 256

typedef struct vtk_pixel_t
{
int c1;
int c2;
int c3;
int c4;	
} vtk_pixel_t;

typedef struct hexahedron_t
{
int c1;
int c2;
int c3;
int c4;
int c5;
int c6;
int c7;
int c8;
} hexahedron_t;

typedef struct coordinates_t
{
float x;
float y;
float z;	
} coordinates_t;


typedef struct cap_t
{
    int id;
    char name[8];
    hid_t group;
} cap_t;


typedef struct field_t
{
    const char *name;
    
    int rank;
    hsize_t *dims;
    hsize_t *maxdims;

    hsize_t *offset;
    hsize_t *count;

    int n;
    float *data;

} field_t;


static cap_t *open_cap(hid_t file_id, int capid);
static herr_t close_cap(cap_t *cap);


static field_t *open_field(cap_t *cap, const char *name);
static herr_t read_field(cap_t *cap, field_t *field, int timestep);
static herr_t close_field(field_t *field);


static herr_t get_attribute_str(hid_t obj_id, const char *attr_name, char **data);
static herr_t get_attribute_int(hid_t input, const char *name, int *val);
static herr_t get_attribute_float(hid_t input, const char *name, float *val);
static herr_t get_attribute(hid_t obj_id, const char *attr_name, hid_t mem_type_id, void *data);
static herr_t get_attribute_mem(hid_t obj_id, const char *attr_name, hid_t mem_type_id, void *data);
static herr_t get_attribute_disk(hid_t loc_id, const char *attr_name, void *attr_out);
static herr_t get_attribute_info(hid_t obj_id, const char *attr_name, hsize_t *dims, H5T_class_t *type_class, size_t *type_size, hid_t *type_id);

static coordinates_t rtf_to_xyz(coordinates_t coord);
static coordinates_t velocity_to_cart(coordinates_t velocity, coordinates_t coord);
static float calc_mean(field_t *topo,int nodey,int nodez);

static void write_vtk_shell(coordinates_t coordinates[], hexahedron_t connectivity[], float temperature[], 
					   float viscosity[], coordinates_t velocity[], int nodex, int nodey, int nodez,
					   int timestep, float radius_inner,int caps,int ascii);


static void write_vtk_surface(coordinates_t coordinates[], vtk_pixel_t connectivity[], 
						float heatflux[], coordinates_t velocity[],
						int timestep,int nodex,int nodey,int nodez,int caps, char* filename_prefix);

int main(int argc, char *argv[])
{
    char *datafile;

    hid_t h5file;
    hid_t input;
    herr_t status;

    int caps;
    int capid;
    cap_t *cap;

    int step;
    int n, i, j, k;
    int nodex, nodey, nodez;
	float radius_inner;
	float radius_outer;
	
	
	int nodex_redu=0;
	int nodey_redu=0;
	int nodez_redu=0;
	int initial=0;
	int timestep=-1;
	
    int current_t=0;
    int timesteps=0;
	
	int cell_counter=0;
	int cell_counter_surface=0;
	
    int *steps;
    char *endptr;

    field_t *coord;
    field_t *velocity;
    field_t *temperature;
    field_t *viscosity;
	
	//Bottom
	field_t *botm_coord;
	field_t *botm_hflux;
	field_t *botm_velo;
	field_t *botm_topo;
	
	//Surface
	field_t *surf_coord;
	field_t *surf_hflux;
	field_t *surf_velo;
	field_t *surf_topo;
	
	///////////////////////////////////////////////
	int bottom=false;
	int surface=false;
	int topo=false;
	int ascii=false;
	
    /************************************************************************
     * Parse command-line parameters.                                       *
     ************************************************************************/

    /*
     * HDF5 file must be specified as first argument.
     */
    
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s file.h5 [step1 [step2 [...]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    char c;
    char *hdf_filepath;
    char *output_filepath;
    extern char *optarg;
    extern int optind, optopt;
	int errflg=0;
    while ((c = getopt(argc, argv, ":p:o:i:t:x:y:z:bscah?")) != -1) {
        switch(c) {
        		case 'p':
					hdf_filepath = optarg;
					break;
        
				case 'o':
            		output_filepath = optarg;
                	printf("Got output filepath\n");
            		break;
								
        		case 'i':
					initial = atoi(optarg);
					printf("Initial: %d\n",initial);
					break;
				
				case 't':
					timesteps = atoi(optarg);
					printf("Timesteps: %d\n", timesteps);
					//Inclusive
					timesteps++;
					break;
				
				case 'x':
					nodex_redu=atoi(optarg);
					break;
				
				case 'y':
					nodey_redu=atoi(optarg);
					break;
				
				case 'z':
					nodez_redu=atoi(optarg);
					break;
				
				
				case ':':       /* missing operand */
                    fprintf(stderr,
                            "Option -%c requires an operand\n", optopt);
                    errflg++;
                	break;
				////////////////////
				
				case 'b':
					bottom=true;
					printf("Create Bottom");
					break;
				
				case 's':
					surface=true;
					printf("Create Surface");
					break;
				
				case 'c':
					topo=true;
					printf("Create Topography");	
					break;
				
				case 'a':
					ascii=true;
					break;
				
				case '?':
            		errflg++;
				break;
        	}
    }
	
    if (errflg==1) {
        fprintf(stderr, "usage: . . . \n");
        exit(2);
    	}
		
    for ( ; optind < argc; optind++) {
        if (access(argv[optind], R_OK)) {
 			printf("Geht\n");
			}
	}

	
	printf("Opening HDF file...\n");
	
    h5file = H5Fopen(hdf_filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    


    /************************************************************************
     * Get mesh parameters.                                                 *
     ************************************************************************/

    /* Read input group */
    input = H5Gopen(h5file, "input");
    if (input < 0)
    {
        fprintf(stderr, "Could not open /input group in \"%s\"\n", argv[1]);
        status = H5Fclose(h5file);
        return EXIT_FAILURE;
    }
	
	
    status = get_attribute_str(input, "datafile", &datafile);
    status = get_attribute_int(input, "nproc_surf", &caps);
    status = get_attribute_int(input, "nodex", &nodex);
    status = get_attribute_int(input, "nodey", &nodey);
    status = get_attribute_int(input, "nodez", &nodez);
	status = get_attribute_float(input,"radius_inner",&radius_inner);
	status = get_attribute_float(input,"radius_outer",&radius_outer);
	
	//Bound input params against hdf
	if(nodex_redu<nodex & nodex_redu>0)
	{
	nodex = nodex_redu;
	}
	if(nodey_redu<nodey & nodey_redu>0)
	{
	nodey = nodey_redu;
	}
	if(nodez_redu<nodez & nodez_redu>0)
	{
	nodez = nodez_redu;
	}
	
	
	printf("Nodex: %d\n",nodex);
	printf("Nodey: %d\n",nodey);
	printf("Nodez: %d\n",nodez);
	printf("Caps: %d\n",caps);
    /* Release input group */
    status = H5Gclose(input);
	

    /************************************************************************
     * Create fields using cap00 datasets as a template.                    *
     ************************************************************************/

    cap         = open_cap(h5file, 0);
    coord       = open_field(cap, "coord");
    velocity    = open_field(cap, "velocity");
    temperature = open_field(cap, "temperature");
    viscosity   = open_field(cap, "viscosity");
    
	
	/*Create fields bottom and surface*/
	botm_coord = open_field(cap,"botm/coord");
	botm_hflux = open_field(cap,"botm/heatflux");
	botm_velo = open_field(cap,"botm/velocity");
	botm_topo = open_field(cap,"botm/topography");
	
	surf_coord = open_field(cap,"surf/coord");
	surf_hflux = open_field(cap,"surf/heatflux");
	surf_velo = open_field(cap,"surf/velocity");
	surf_topo = open_field(cap,"surf/topography");
	
	status      = close_cap(cap);

	
    /************************************************************************
     * Output requested data.                                               *
     ************************************************************************/
	int iterations=0;
    /* Iterate over timesteps */
    for(current_t = initial; current_t < timesteps; current_t++)
    {
		
		printf("\nProcessing timestep: %d\n",current_t);
      
		
		coordinates_t ordered_coordinates[((nodex*nodey*nodez)*caps)];
		coordinates_t ordered_velocity[((nodex*nodey*nodez)*caps)*3];		
		float ordered_temperature[(nodex*nodey*nodez)*caps];
		float ordered_viscosity[(nodex*nodey*nodez)*caps];
		hexahedron_t connectivity[((nodex-1)*(nodey-1)*(nodez-1))*caps];
		
		
		coordinates_t ordered_botm_coords[(nodex*nodey*caps)];
		float ordered_botm_hflux[(nodex*nodey*caps)];
		coordinates_t ordered_botm_velocity[(nodex*nodey*caps)];
		
		
		coordinates_t ordered_surf_coords[(nodex*nodey*caps)];
		float ordered_surf_hflux[(nodex*nodey*caps)];
		coordinates_t ordered_surf_velocity[(nodex*nodey*caps)];
		
		vtk_pixel_t connectivity_surface[(nodex*nodey*caps)];
		
		//Holds single coordinate		
		coordinates_t coordinate;
		
		//Holds single vector
		coordinates_t velocity_vector;
				
        /* Iterate over caps */
		
		for(capid = 0; capid < caps; capid++)
        {
            cap = open_cap(h5file, capid);
			printf("Processing cap %d of %d\n",capid+1,caps);
            //snprintf(filename, (size_t)99, "%s.cap%02d.%d", datafile, capid, step);
            //fprintf(stderr, "Writing %s\n", filename);

            //file = fopen(filename, "w");
            //fprintf(file, "%d x %d x %d\n", nodex, nodey, nodez);

            /* Read data from HDF5 file. */
            read_field(cap, coord, 0);
            read_field(cap, velocity, current_t);
            read_field(cap, temperature, current_t);
            read_field(cap, viscosity, current_t);
			
			
			
    		/*Counts iterations*/
    		n = 0;
			
			//Number of nodes per cap
			int nodes=nodex*nodey*nodez;
			
	        /* Traverse data in Citcom order */
            for(j = 0; j < nodey; j++)
            {	
                for(i = 0; i < nodex; i++)
                {
			       for(k = 0; k < nodez; k++)
                    {       
								//Coordinates						
								coordinate.x = coord->data[3*n+0];
                                coordinate.y = coord->data[3*n+1];
                                coordinate.z = coord->data[3*n+2];
								coordinate = rtf_to_xyz(coordinate);
								ordered_coordinates[n+(capid*nodes)].x = coordinate.x;
								ordered_coordinates[n+(capid*nodes)].y = coordinate.y;
								ordered_coordinates[n+(capid*nodes)].z = coordinate.z;
								
								//Velocity
                        	    velocity_vector.x = velocity->data[3*n+0];
                                velocity_vector.y = velocity->data[3*n+1];
                                velocity_vector.z = velocity->data[3*n+2];
						
								velocity_vector = velocity_to_cart(velocity_vector,coordinate);
								
								ordered_velocity[n+(capid*nodes)].x = velocity_vector.x;
								ordered_velocity[n+(capid*nodes)].y = velocity_vector.y;
								ordered_velocity[n+(capid*nodes)].z = velocity_vector.z;
						
								//Temperature
                                ordered_temperature[n+(capid*nodes)] = temperature->data[n];
						
								//Viscosity
                                ordered_viscosity[n+(capid*nodes)] = viscosity->data[n];
								
								n++;
                    }
                }
            }

			//Create connectivity
			if(iterations==0)
			{
				//For 3d Data 
            	int i=1;    //Counts X Direction
            	int j=1;    //Counts Y Direction
            	int k=1;    //Counts Z Direction
    		
            	for(n=0; n<((nodex*nodey*nodez)-(nodex*nodez));n++)
					{
						
                		if ((i%nodez)==0)   //X-Values
							{
                    		j++;                 //Count Y-Values
        					}
                		if ((j%nodex)==0)
							{
                    		k++;                 //Count Z-Values
                  			}
							
                		if (((i%nodez) != 0) && ((j%nodex) != 0))            //Check if Box can be created
							{
							//Create Connectivity
                    		connectivity[cell_counter].c1 = n+(capid*(nodes));
							connectivity[cell_counter].c2 = connectivity[cell_counter].c1+nodez;
                    		connectivity[cell_counter].c3 = connectivity[cell_counter].c2+nodez*nodex;
                    		connectivity[cell_counter].c4 = connectivity[cell_counter].c1+nodez*nodex;
                    		connectivity[cell_counter].c5 = connectivity[cell_counter].c1+1;
                    		connectivity[cell_counter].c6 = connectivity[cell_counter].c5+nodez;
                    		connectivity[cell_counter].c7 = connectivity[cell_counter].c6+nodez*nodex;
                    		connectivity[cell_counter].c8 = connectivity[cell_counter].c5+nodez*nodex;
							cell_counter++;
							}                   	
                i++;
				
      			}
			}
			

			
			
			//Bottom and Surface
			
			if(bottom==true){
				
				/*Read Bottom data from HDF5 file.*/
				read_field(cap,botm_coord,0);
				read_field(cap,botm_hflux,current_t);
				read_field(cap,botm_velo,current_t);
				read_field(cap,botm_topo,current_t);
				float botm_mean=0.0;	
				if(topo=true)
					{
					botm_mean = calc_mean(botm_topo,nodex,nodey);
					}					
				for(n=0;n<nodex*nodey;n++)
				{
				//Coordinates						
				coordinate.x = botm_coord->data[2*n+0];
    			coordinate.y = botm_coord->data[2*n+1];
    				if(topo==true)
					{
					coordinate.z = radius_inner+(botm_topo->data[n]-botm_mean)*
						(pow(10.0,21.0)/(pow(6371000,2)/pow(10,-6))/3300*10)/1000;
					//printf("Z: %f\n",coordinate.z);
					}
					else
					{
					coordinate.z = radius_inner;
					}
					
				coordinate = rtf_to_xyz(coordinate);
				ordered_botm_coords[n+(capid*nodex*nodey)].x = coordinate.x;
				ordered_botm_coords[n+(capid*nodex*nodey)].y = coordinate.y;
				ordered_botm_coords[n+(capid*nodex*nodey)].z = coordinate.z;
								
				ordered_botm_hflux[n+((capid)*nodex*nodey)] = botm_hflux->data[n];
					
				velocity_vector.x = botm_velo->data[3*n+0];
				velocity_vector.y = botm_velo->data[3*n+1];
				velocity_vector.z = botm_velo->data[3*n+2];
					
				velocity_vector = velocity_to_cart(velocity_vector,coordinate);
					
				ordered_botm_velocity[n+(capid*nodex*nodey)].x = velocity_vector.x;
				ordered_botm_velocity[n+(capid*nodex*nodey)].y = velocity_vector.y;
				ordered_botm_velocity[n+(capid*nodex*nodey)].z = velocity_vector.z;
								
				}		
	
	
			}
			
			if(surface==true)
			{
		
				/*Read Surface data from HDF5 file.*/
				read_field(cap,surf_coord,0);
				read_field(cap,surf_hflux,current_t);
				read_field(cap,surf_velo,current_t);
				read_field(cap,surf_topo,current_t);
				float surf_mean=0.0;
				if(topo=true)
					{
						
					surf_mean = calc_mean(surf_topo,nodex,nodey);
					}					
				for(n=0;n<nodex*nodey;n++)
				{
				//Coordinates						
				coordinate.x = surf_coord->data[2*n+0];
    			coordinate.y = surf_coord->data[2*n+1];
    				if(topo==true)
					{
					coordinate.z = radius_outer+(surf_topo->data[n]-surf_mean)*
						(pow(10.0,21.0)/(pow(6371000,2)/pow(10,-6))/3300*10)/1000;
					//printf("Z: %f\n",coordinate.z);
					}
					else
					{
					coordinate.z = radius_outer;
					}
					
				coordinate = rtf_to_xyz(coordinate);
				ordered_surf_coords[n+(capid*nodex*nodey)].x = coordinate.x;
				ordered_surf_coords[n+(capid*nodex*nodey)].y = coordinate.y;
				ordered_surf_coords[n+(capid*nodex*nodey)].z = coordinate.z;
								
				ordered_surf_hflux[n+((capid)*nodex*nodey)] = botm_hflux->data[n];
					
				velocity_vector.x = botm_velo->data[3*n+0];
				velocity_vector.y = botm_velo->data[3*n+1];
				velocity_vector.z = botm_velo->data[3*n+2];
					
				velocity_vector = velocity_to_cart(velocity_vector,coordinate);
					
				ordered_surf_velocity[n+(capid*nodex*nodey)].x = velocity_vector.x;
				ordered_surf_velocity[n+(capid*nodex*nodey)].y = velocity_vector.y;
				ordered_surf_velocity[n+(capid*nodex*nodey)].z = velocity_vector.z;
								
				}		
				
			}

			
			//Create connectivity information 2d
			
			
			if(iterations==0){
				if(surface==true | bottom==true)
					{
					
            		for(n=0;n<(nodex*nodey)-nodey;n++)
						{
                    	if ((n+1)%nodey!=0){
							connectivity_surface[cell_counter_surface].c1 = n+(capid*((nodex)*(nodey)));
                        	connectivity_surface[cell_counter_surface].c2 = connectivity_surface[cell_counter_surface].c1+1;
							connectivity_surface[cell_counter_surface].c3 = connectivity_surface[cell_counter_surface].c1+nodey;
							connectivity_surface[cell_counter_surface].c4 = connectivity_surface[cell_counter_surface].c3+1;          
							cell_counter_surface++;
							}
						}
					}
				}
			
			close_cap(cap);
        }
		
    iterations++;
		
	//Write data to file
	write_vtk_shell(ordered_coordinates, connectivity, ordered_temperature, 
					   ordered_viscosity, ordered_velocity, nodex, nodey, nodez,
					   current_t, radius_inner,caps,ascii);
	
	if(bottom==true){
		write_vtk_surface(ordered_botm_coords,connectivity_surface,ordered_botm_hflux,
						  ordered_botm_velocity,current_t,nodex,nodey,nodez,caps,"bottom");
	}
	
	
	if(surface==true){
		write_vtk_surface(ordered_surf_coords,connectivity_surface,ordered_surf_hflux,
						  ordered_surf_velocity,current_t,nodex,nodey,nodez,caps,"surface");
	}
		
	}//end timesteps loop

    /* Release resources. */

    status = close_field(coord);
    status = close_field(velocity);
    status = close_field(temperature);
    status = close_field(viscosity);
    status = H5Fclose(h5file);

    return EXIT_SUCCESS;
}


static cap_t *open_cap(hid_t file_id, int capid)
{
    cap_t *cap;
    cap = (cap_t *)malloc(sizeof(cap_t));
    cap->id = capid;
    snprintf(cap->name, (size_t)7, "cap%02d", capid);
    cap->group = H5Gopen(file_id, cap->name);
    if (cap->group < 0)
    {
        free(cap);
        return NULL;
    }
    return cap;
}


static herr_t close_cap(cap_t *cap)
{
    herr_t status;
    if (cap != NULL)
    {
        cap->id = -1;
        cap->name[0] = '\0';
        status = H5Gclose(cap->group);
        free(cap);
    }
    return 0;
}


static field_t *open_field(cap_t *cap, const char *name)
{
    hid_t dataset;
    hid_t dataspace;
    herr_t status;

    int d;
    int rank;

    field_t *field;

    if (cap == NULL)
        return NULL;


    /* Allocate field and initialize. */

    field = (field_t *)malloc(sizeof(field_t));

    field->name = name;
    field->rank = 0;
    field->dims = NULL;
    field->maxdims = NULL;
    field->n = 0;

    dataset = H5Dopen(cap->group, name);
    if(dataset < 0)
    {
        free(field);
        return NULL;
    }

    dataspace = H5Dget_space(dataset);
    if (dataspace < 0)
    {
        free(field);
        return NULL;
    }


    /* Calculate shape of field. */

    rank = H5Sget_simple_extent_ndims(dataspace);

    field->rank = rank;
    field->dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
    field->maxdims = (hsize_t *)malloc(rank * sizeof(hsize_t));

    status = H5Sget_simple_extent_dims(dataspace, field->dims, field->maxdims);

    /* DEBUG
    printf("Field %s shape (", name);
    for(d = 0; d < rank; d++)
        printf("%d,", (int)(field->dims[d]));
    printf(")\n");
    // */


    /* Allocate memory for hyperslab selection parameters. */

    field->offset = (hsize_t *)malloc(rank * sizeof(hsize_t));
    field->count  = (hsize_t *)malloc(rank * sizeof(hsize_t));


    /* Allocate enough memory for a single time-slice buffer. */

    field->n = 1;
    if (field->maxdims[0] == H5S_UNLIMITED)
        for(d = 1; d < rank; d++)
            field->n *= field->dims[d];
    else
        for(d = 0; d < rank; d++)
            field->n *= field->dims[d];

    field->data = (float *)malloc(field->n * sizeof(float));


    /* Release resources. */

    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);

    return field;
}


static herr_t read_field(cap_t *cap, field_t *field, int timestep)
{
    hid_t dataset;
    hid_t filespace;
    hid_t memspace;
    herr_t status;

    int d;

    if (cap == NULL || field == NULL)
        return -1;

    dataset = H5Dopen(cap->group, field->name);

    if (dataset < 0)
        return -1;

    for(d = 0; d < field->rank; d++)
    {
        field->offset[d] = 0;
        field->count[d]  = field->dims[d];
    }

    if (field->maxdims[0] == H5S_UNLIMITED)
    {
        field->offset[0] = timestep;
        field->count[0]  = 1;
    }

    /* DEBUG
    printf("Reading step %d on field %s with offset (", timestep, field->name);
    for(d = 0; d < field->rank; d++) printf("%d,", (int)(field->offset[d]));
    printf(") and count (");
    for(d = 0; d < field->rank; d++) printf("%d,", (int)(field->count[d]));
    printf(")\n");
    // */


    filespace = H5Dget_space(dataset);

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 field->offset, NULL, field->count, NULL);

    memspace = H5Screate_simple(field->rank, field->count, NULL);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT, memspace,
                     filespace, H5P_DEFAULT, field->data);
    
    status = H5Sclose(filespace);
    status = H5Sclose(memspace);
    status = H5Dclose(dataset);

    return 0;
}


static herr_t close_field(field_t *field)
{
    if (field != NULL)
    {
        free(field->dims);
        free(field->maxdims);
        free(field->offset);
        free(field->count);
        free(field->data);
        free(field);
    }
    return 0;
}


static herr_t get_attribute_str(hid_t obj_id,
                                const char *attr_name,
                                char **data)
{
    hid_t attr_id;
    hid_t attr_type;
    size_t type_size;
    herr_t status;

    *data = NULL;

    attr_id = H5Aopen_name(obj_id, attr_name);
    if (attr_id < 0)
        return -1;

    attr_type = H5Aget_type(attr_id);
    if (attr_type < 0)
        goto out;

    /* Get the size */
    type_size = H5Tget_size(attr_type);
    if (type_size < 0)
        goto out;

    /* malloc enough space for the string, plus 1 for trailing '\0' */
    *data = (char *)malloc(type_size + 1);

    status = H5Aread(attr_id, attr_type, *data);
    if (status < 0)
        goto out;

    /* Set the last character to '\0' in case we are dealing with
     * null padded or space padded strings
     */
    (*data)[type_size] = '\0';

    status = H5Tclose(attr_type);
    if (status < 0)
        goto out;

    status = H5Aclose(attr_id);
    if (status < 0)
        return -1;

    return 0;

out:
    H5Tclose(attr_type);
    H5Aclose(attr_id);
    if (*data)
        free(*data);
    return -1;
}


static herr_t get_attribute_int(hid_t input, const char *name, int *val)
{
    hid_t attr_id;
    hid_t type_id;
    H5T_class_t type_class;
    size_t type_size;

    herr_t status;

    char *strval;

    attr_id = H5Aopen_name(input, name);
    type_id = H5Aget_type(attr_id);
    type_class = H5Tget_class(type_id);
    type_size = H5Tget_size(type_id);

    H5Tclose(type_id);
    H5Aclose(attr_id);

    switch(type_class)
    {
        case H5T_STRING:
            status = get_attribute_str(input, name, &strval);
            if (status < 0) return -1;
            *val = atoi(strval);
            free(strval);
            return 0;
        case H5T_INTEGER:
            status = get_attribute(input, name, H5T_NATIVE_INT, val);
            if (status < 0) return -1;
            return 0;
    }

    return -1;
}

static herr_t get_attribute_float(hid_t input, const char *name, float *val)
{
    hid_t attr_id;
    hid_t type_id;
    H5T_class_t type_class;
    size_t type_size;

    herr_t status;

    char *strval;

    attr_id = H5Aopen_name(input, name);
    type_id = H5Aget_type(attr_id);
    type_class = H5Tget_class(type_id);
    type_size = H5Tget_size(type_id);

    H5Tclose(type_id);
    H5Aclose(attr_id);

    switch(type_class)
    {
        case H5T_STRING:
            status = get_attribute_str(input, name, &strval);
            if (status < 0) return -1;
                *val = atof(strval);
            free(strval);
            return 0;
        case H5T_FLOAT:
            status = get_attribute(input, name, H5T_NATIVE_FLOAT, val);
            if (status < 0) return -1;
            return 0;
    }
    return -1;
}


static herr_t get_attribute(hid_t obj_id,
                            const char *attr_name,
                            hid_t mem_type_id,
                            void *data)
{
    herr_t status;

    status = get_attribute_mem(obj_id, attr_name, mem_type_id, data);
    if (status < 0)
        return -1;

    return 0;
}


static herr_t get_attribute_mem(hid_t obj_id,
                                const char *attr_name,
                                hid_t mem_type_id,
                                void *data)
{
    hid_t attr_id;
    herr_t status;

    attr_id = H5Aopen_name(obj_id, attr_name);
    if (attr_id < 0)
        return -1;

    status = H5Aread(attr_id, mem_type_id, data);
    if (status < 0)
    {
        H5Aclose(attr_id);
        return -1;
    }

    status = H5Aclose(attr_id);
    if (status < 0)
        return -1;

    return 0;
}


static herr_t get_attribute_disk(hid_t loc_id,
                                 const char *attr_name,
                                 void *attr_out)
{
    hid_t attr_id;
    hid_t attr_type;
    herr_t status;

    attr_id = H5Aopen_name(loc_id, attr_name);
    if (attr_id < 0)
        return -1;

    attr_type = H5Aget_type(attr_id);
    if (attr_type < 0)
        goto out;

    status = H5Aread(attr_id, attr_type, attr_out);
    if (status < 0)
        goto out;

    status = H5Tclose(attr_type);
    if (status < 0)
        goto out;

    status = H5Aclose(attr_id);
    if (status < 0)
        return -1;

    return 0;
out:
    H5Tclose(attr_type);
    H5Aclose(attr_id);
    return -1;
}


static herr_t get_attribute_info(hid_t obj_id,
                                 const char *attr_name,
                                 hsize_t *dims,
                                 H5T_class_t *type_class,
                                 size_t *type_size,
                                 hid_t *type_id)
{
    hid_t attr_id;
    hid_t space_id;
    herr_t status;
    int rank;

    /* Open the attribute. */
    attr_id = H5Aopen_name(obj_id, attr_name);
    if (attr_id < 0)
        return -1;

    /* Get an identifier for the datatype. */
    *type_id = H5Aget_type(attr_id);

    /* Get the class. */
    *type_class = H5Tget_class(*type_id);

    /* Get the size. */
    *type_size = H5Tget_size(*type_id);

    /* Get the dataspace handle */
    space_id = H5Aget_space(attr_id);
    if (space_id < 0)
        goto out;

    /* Get dimensions */
    rank = H5Sget_simple_extent_dims(space_id, dims, NULL);
    if (rank < 0)
        goto out;

    /* Terminate access to the dataspace */
    status = H5Sclose(space_id);
    if (status < 0)
        goto out;

    /* End access to the attribute */
    status = H5Aclose(attr_id);
    if (status < 0)
        goto out;

    return 0;
out:
    H5Tclose(*type_id);
    H5Aclose(attr_id);
    return -1;
}



static coordinates_t rtf_to_xyz(coordinates_t coord)
{
	coordinates_t output;
	output.x = coord.z * sin(coord.x) * cos(coord.y);
    output.y = coord.z * sin(coord.x) * sin(coord.y);
    output.z = coord.z * cos(coord.x);
	return output;
}


static coordinates_t velocity_to_cart(coordinates_t velocity, coordinates_t coord)
{
    
	coordinates_t output;
	output.x = velocity.z*sin(coord.x)*cos(coord.y)+velocity.x*cos(coord.x)*cos(coord.y)-velocity.y*sin(coord.y);
    output.y = velocity.z*sin(coord.x)*sin(coord.y)+velocity.x*cos(coord.x)*sin(coord.y)+velocity.y*cos(coord.y);
    output.z = velocity.z*cos(coord.x)-velocity.x*sin(coord.x);
    return output;
}

static void write_vtk_shell(coordinates_t coordinates[], hexahedron_t connectivity[], float temperature[], 
						float viscosity[], coordinates_t velocity[], int nodex, int nodey, int nodez,
						int timestep, float radius_inner,int caps,int ascii)
{
	FILE *file;
    char filename[100];
    char header[STRING_BUFFER];
	char puffer;
	int i;
	snprintf(filename, (size_t)99, "%s.%d.vtk", "datafile", timestep);
    fprintf(stderr, "Writing %s\n", filename);
	file = fopen(filename, "w");
	int nodes = nodex*nodey*nodez;
	
	char *type_info;
	
	if(ascii==true)
	{
	type_info = "Ascii";
	}
	else
	{
	type_info = "Binary";
	}
	//Write Header
	sprintf(header,"# vtk DataFile Version 2.0\n\
CitcomS Output Timestep:%d NX:%d NY:%d NZ:%d Radius_Inner:%f\n\
%s\n\
DATASET UNSTRUCTURED_GRID\n",timestep,nodex,nodey,nodez,radius_inner,ascii	?  "ASCII" : "BINARY");
	fprintf(file,header);
	
	
	fprintf(file, "POINTS %d float\n",nodes*caps);
	//printf("Float: %d\n",sizeof(float));
	//printf("Field: %d\n",sizeof(coordinates_t));
	
	//Write Coordinates
	if(ascii==true)
		{
		for(i=0;i<nodes*caps;i++)
			{
			fprintf(file, "%f %f %f\n",coordinates[i].x,coordinates[i].y,coordinates[i].z);
			}
		}
		else
		{
		//fwrite(&coordinates,sizeof(coordinates_t),nodes*caps,file);
			
		for(i=0;i<nodes*caps;i++){
			
			fwrite(&coordinates[i].x,sizeof(float),1,file);
			fwrite(&coordinates[i].y,sizeof(float),1,file);			
			fwrite(&coordinates[i].z,sizeof(float),1,file);			
			}
		}
		
	int cells = ((nodex-1)*(nodey-1)*(nodez-1))*caps;
	//Write Cells
	fprintf(file, "CELLS %d %d\n",cells,cells*9);
	for(i=0;i<cells;i++)
		{
		fprintf(file, "8 %d %d %d %d %d %d %d %d\n",
					connectivity[i].c1,
					connectivity[i].c2,
					connectivity[i].c3,
					connectivity[i].c4,
					connectivity[i].c5,
					connectivity[i].c6,
					connectivity[i].c7,
					connectivity[i].c8	
					);
		}
		
	//Write Cell Types Hexahedron
	
		fprintf(file,"CELL_TYPES %d\n",cells);
		int j=0;
		for(i=0;i<cells;i++)
		{
		fprintf(file,"12 ");
		j++;
		if(j==8)				//nicer formating
			{
			fprintf(file,"\n");
			j=0;
			}
	
		}
		
		
		
	//Write Scalar Temperature
	fprintf(file,"POINT_DATA %d\n",nodes*caps);
	fprintf(file,"SCALARS Temperature_scalars float 1\n");
	fprintf(file,"LOOKUP_TABLE default\n");
	
	
	for(i=0;i<nodes*caps;i++)
		{
		fprintf(file,"%f ", temperature[i]);
		
		if((i+1)%nodex==0)
			{
			fprintf(file,"\n");
			
			}			
		}

	
	//Write Scalar Viscosity
	fprintf(file,"SCALARS Viscosity_scalars float 1\n");
	fprintf(file,"LOOKUP_TABLE default\n");
	
	
	for(i=0;i<nodes*caps;i++)
		{
		fprintf(file,"%f ", viscosity[i]);
		
		if((i+1)%nodex==0)
			{
			fprintf(file,"\n");
			
			}			
		}

	
	//Write Velocity Vectors
	fprintf(file,"Vectors Velocity_vectors float\n");
	
	
	for(i=0;i<nodes*caps;i++)
		{
		fprintf(file,"%f %f %f \n", velocity[i].x, velocity[i].y, velocity[i].z);
		}
		
		

	fclose(file);
		
}

static void write_vtk_surface(coordinates_t coordinates[], vtk_pixel_t connectivity[], 
						float heatflux[], coordinates_t velocity[],
						int timestep,int nodex,int nodey,int nodez,int caps, char* filename_prefix)
{
	FILE *file;
    char filename[100];
    
	
	snprintf(filename, (size_t)99, "%s.%d.vtk", filename_prefix, timestep);
    fprintf(stderr, "Writing %s\n", filename);
	file = fopen(filename, "w");
    int i;
	int nodes = nodex*nodey;

	//Write Header	
	fprintf(file,"# vtk DataFile Version 2.0\n");
	fprintf(file,"CitcomS Output %s Timestep:%d NX:%d NY:%d NZ:%d\n",
				filename_prefix,timestep,nodex,nodey,nodez);
	
	fprintf(file, "ASCII\n");
	fprintf(file, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(file, "POINTS %d float\n",nodes*caps);
	
	//Write Coordinates
	for(i=0;i<nodes*caps;i++)
		{
		fprintf(file, "%f %f %f\n",coordinates[i].x,coordinates[i].y,coordinates[i].z);
		}

	//Write Cells
	int cells = ((nodex-1)*(nodey-1))*caps;
	fprintf(file, "CELLS %d %d\n",cells,cells*5);
	for(i=0;i<(((nodex-1)*(nodey-1))*caps);i++)
		{
		fprintf(file, "4 %d %d %d %d\n",
					connectivity[i].c1,
					connectivity[i].c2,
					connectivity[i].c3,
					connectivity[i].c4
					);
		}
	
	//Write Cell Types Pixel
	
		fprintf(file,"CELL_TYPES %d\n",cells);
		int j=0;
		for(i=0;i<cells;i++)
		{
		fprintf(file,"8 ");
		j++;
		if(j==8)				//nicer formating
			{
			fprintf(file,"\n");
			j=0;
			}
	
		}
	
		
		
	//Write Scalar Temperature
	fprintf(file,"POINT_DATA %d\n",nodes*caps);
	fprintf(file,"SCALARS Temperature_scalars float 1\n");
	fprintf(file,"LOOKUP_TABLE default\n");
	
	
	for(i=0;i<nodes*caps;i++)
		{
		fprintf(file,"%f ", heatflux[i]);
		
		if((i+1)%nodex==0)
			{
			fprintf(file,"\n");
			}			
		}

	//Write Velocity Vectors
	fprintf(file,"Vectors Velocity_vectors float\n");
	
	
	for(i=0;i<nodes*caps;i++)
		{
		fprintf(file,"%f %f %f \n", velocity[i].x, velocity[i].y, velocity[i].z);
		}

	fclose(file);
}

static float calc_mean(field_t *topo,int nodex,int nodey)
{
int i;
int nodes = (nodex*nodey);
double f;
for(i=0;i<nodes;i++)
	{
	f = f+topo->data[i];
	}
return (float) f/nodes;
}
