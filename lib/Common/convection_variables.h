struct CONVECTION { /* information controlling convection problems */
    char old_T_file[100];

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

} convection;


