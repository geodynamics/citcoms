
FILE* output_init(char *output_file);
void output_close(FILE* fp1);
void output_coord_header(struct All_variables *E, FILE *fp1);
void output_coord(struct All_variables *E, FILE *fp1);
void output_velo_header(struct All_variables *E, FILE *fp1, int file_number);
void output_velo(struct All_variables *E, FILE *fp1);
