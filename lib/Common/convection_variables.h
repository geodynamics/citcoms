struct CONVECTION { /* information controlling convection problems */   
    char old_T_file[100];
   
    float temp_blob_x[40];
    float temp_blob_y[40];
    float temp_blob_z[40];
    float temp_blob_radius[40]; /* +/- */
    float temp_blob_T[40];
    float temp_blob_bg[40];     /* Reference level if sticky */
    int temp_blob_sticky[40];
    int temp_blobs;
 
    float temp_zonex1[40];
    float temp_zonex2[40];
    float temp_zonez1[40];
    float temp_zonez2[40];
    float temp_zoney1[40];
    float temp_zoney2[40];
    float temp_zonehw[40];
    float temp_zonemag[40];
    int temp_zone_sticky[40];
    int temp_zones;

    float half_space_age;
    int half_space_cooling;

    int number_of_perturbations;
    int perturb_ll[32];
    int perturb_mm[32];
    int load_depth[32];
    float perturb_mag[32];

    struct SOURCES {
	    int number;
	    float t_offset;
	    float Q[10];
	    float lambda[10];
	}  heat_sources;

    float elasticity1;
  
} convection;


