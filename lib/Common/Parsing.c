/* Routines which read filenames from the command line and
   then parse the contents as parameters for citcom */

#include <stdio.h>
#include <sys/types.h>
#ifndef __sunos__
#include <strings.h>
#else
#include <string.h>
#endif
#include "global_defs.h"

#define MAXLINE		1024	/* max length of line in input file */
#define MAXNAME		64	/* max length of name */
#define MAXVALUE	1024	/* max length of value */
#define MAXFILENAME	64	/* max length of par file name */
#define MAXVECTOR	10	/* max # of elements for unspecified vectors */
#define STRANGE_NUM	-98765.4321

/* abbreviations: */
#define AL 		struct arglist
#define PROGNAME	ext_par.progname
#define FLAGS		ext_par.argflags
#define ARGLIST		ext_par.arglist
#define ARGHEAD		ext_par.arghead
#define ARGBUF		ext_par.argbuf
#define NLIST		ext_par.nlist
#define NBUF		ext_par.nbuf
#define LISTMAX		ext_par.listmax
#define BUFMAX		ext_par.bufmax
#define LISTFILE	ext_par.listout

#define LISTINC		32	/* increment size for arglist */
#define BUFINC		1024	/* increment size for argbuf */

struct ext_par		/* global variables for getpar */
{
  char *progname;
  int argflags;
  struct arglist *arglist;
  struct arglist *arghead;
  char *argbuf;
  int nlist;
  int nbuf;
  int listmax;
  int bufmax;
  FILE *listout;
}	ext_par;

struct arglist		/* structure of list set up by setpar */
{
    int argname_offset;
    int argval_offset;
    int hash;
};

int VERBOSE = 0;
int DESCRIBE = 0;
int BEGINNER = 0;

void setup_parser(E,filename)
     struct All_variables *E;
     char *filename;     
{
    void unique_copy_file();
    
    FILE * fp;
    char *pl,*pn,*pv;
    char t1, t2, line[MAXLINE], name[MAXNAME], value[MAXVALUE];
    int i,j,k;
    int m=E->parallel.me;
  
    /* should get file length & cpp &c before any further parsing */

    /* for now, read one filename from the command line, we'll parse that ! */
    

	// this section moved to main()
/*     if (ac < 2)   { */
/* 	fprintf(stderr,"Usage: citcom PARAMETERFILE\n"); */
/* 	exit(10); */
/*     } */
 

    if ((fp = fopen(filename,"r")) == NULL)  {
      fprintf(stderr,"(Parsing #1) File: %s is unreadable\n",filename);
      exit(11);
    }

    unique_copy_file(E,filename,"copy");


  /* now the parameter file is open, read into memory */

  while( fgets(line,MAXLINE,fp) != NULL )
    { pl= line;
      /* loop over entries on each line */
    loop:	
      while(*pl==' ' || *pl=='\t') pl++;
      if(*pl=='\0'|| *pl=='\n') continue; /* end of line */
      if(*pl=='#') continue; /* end of interpretable part of line */

      /* get name */
      pn= name;
      while(*pl != '=' && *pl != '\0' && *pl != ' '
	    && *pl != '\n'		/* FIX by Glenn Nelson */
	    && *pl != '\t') 
	*pn++ = *pl++;
      *pn = '\0';
      if(*pl == '=') pl++;
      
      /* get value */
      *value= '\0';
      pv= value;
      if(*pl=='"' || *pl=='\'')
	t1= t2= *pl++; 
      else
	{ t1= ' ';
	  t2= '\t';
	}
      while(*pl!=t1 && *pl!=t2 &&
	    *pl!='\0' && *pl!='\n') *pv++= *pl++;
      *pv= '\0';
      if(*pl=='"' || *pl=='\'')
	pl++;
      add_to_parameter_list(name,value);
     
      goto loop;
    }

  fclose(fp);

  ARGHEAD= ARGLIST;

  /* Now we can use our routines to check & set their own flags ! */

  input_boolean("VERBOSE",E->control.VERBOSE,"off",m);
  input_boolean("DESCRIBE",E->control.DESCRIBE,"off",m);
  input_boolean("BEGINNER",E->control.BEGINNER,"off",m);
  VERBOSE=E->control.VERBOSE;
  DESCRIBE=E->control.DESCRIBE;
  BEGINNER=E->control.BEGINNER;
  
}

void shutdown_parser(E)
     struct All_variables *E;

{
	if(ARGLIST != NULL) free(ARGLIST);
	if(ARGBUF  != NULL) free(ARGBUF);
	ARGBUF=  NULL;
	ARGLIST= NULL;
	
}


add_to_parameter_list(name,value)	/* add an entry to arglist, expanding memory */
     register char *name, *value;	/* if necessary */
{
  struct arglist *alptr;
  int len;
  register char *ptr;

  /* check arglist memory */
  if(NLIST >= LISTMAX)
    { LISTMAX += LISTINC;
      if(ARGLIST == NULL)
	ARGLIST= (AL *)malloc(LISTMAX * sizeof(AL));
      else	
	ARGLIST= (AL *)realloc(ARGLIST,LISTMAX * sizeof(AL));
    }
  /* check argbuf memory */
  len= strlen(name) + strlen(value) + 2; /* +2 for terminating nulls */
  if(NBUF+len >= BUFMAX)
    { BUFMAX += BUFINC;
      if(ARGBUF == NULL)
	ARGBUF= (char *)malloc(BUFMAX);
      else	ARGBUF= (char *)realloc(ARGBUF,BUFMAX);
    }
  if(ARGBUF == NULL || ARGLIST == NULL)
   fprintf(stderr,"cannot allocate memory\n");

  /* add name */
  alptr= ARGLIST + NLIST;
  alptr->hash= compute_parameter_hash_table(name);
  alptr->argname_offset = NBUF;
  ptr= ARGBUF + NBUF;
  do 
    *ptr++ = *name; 
  while(*name++);
  
  /* add value */
  NBUF += len;
  alptr->argval_offset= ptr - ARGBUF;
  do
    *ptr++ = *value;
  while(*value++);
  NLIST++;
}

int compute_parameter_hash_table(s)
     register char *s;
{ register int h;
  
  h= s[0];
  if(s[1])
    h |= (s[1])<<8;
  else
    return(h);
  if(s[2])
    h |= (s[2])<<16;
  else 
    return(h);
  if(s[3])
    h |= (s[3])<<24;
  return(h);
}

int input_int(name,value,interpret,m)
     char *name;
     int *value;
     char *interpret;
     int m;

{
    int interpret_control_string();
    struct arglist *alptr; 
    int h, found;
    char  *str;
    
  int exists,essential;
  double Default,minvalue,maxvalue;

  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_int: searching for '%s' with default/range '%s'\n",
	    name,(interpret == NULL) ? "**EMPTY**" : interpret);
 
  exists = interpret_control_string(interpret,&essential,&Default,&minvalue,&maxvalue);
 
  *value = (int)(Default);
 
  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;

      str= ARGBUF + alptr->argval_offset;
      sscanf(str,"%d",value);
      found=1;
      break;
    } 

  if(essential && !found)
    { fprintf(stderr,"There MUST be an entry for the parameter %s\n",name);
      exit(12);
    }
  if((minvalue!=STRANGE_NUM) && (*value < (int) minvalue))
     { *value = (int) minvalue;
     }
  if((maxvalue!=STRANGE_NUM) && (*value > (int) maxvalue))
    {  *value = (int) maxvalue;
    }

  if(m==0)
  if(VERBOSE)
   { if (found)
       fprintf(stderr,"%25s: (int) = %d \n",name,*value); 
     else
       if (Default != STRANGE_NUM)
	  fprintf(stderr,"%25s: (int) = not found (%d) \n",name,(int)(Default)); 
       else
	 { fprintf(stderr,"%25s: (int) = not found (no default) \n",name); 
	   if(BEGINNER)
	     { fprintf(stderr,"\t\t Previously set value gives ...");
	       fprintf(stderr,"%d\n",*value);
	     }
	  } 
   }
  
  return(found);
}

int input_string(name,value,Default,m)  /* in the case of a string default=NULL forces input */
     char *name;
     char *value;
     char *Default;
     int m;
{ 
    char *sptr;
  struct arglist *alptr; 
  int h, hno, hyes, found;
  char line[MAXLINE], *str, *noname;
  int essential;

 
  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_string: searching for '%s' with default '%s'\n",
	    name,(Default == NULL) ? "no default" : Default);
 
  h=compute_parameter_hash_table(name);
  essential=found=0;

    
    if (Default != NULL)   /* Cannot use "Essential" as this is a valid input */
	strcpy(value,Default);  
    else
	essential=1;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;

      str= ARGBUF + alptr->argval_offset;
      strcpy(value,str);
      found=1;
      break;
    } 
  
  if(essential && !found)
    { fprintf(stderr,"There MUST be an entry for the parameter %s\n",name);
      exit(12);
    }
 
  if(m==0)
  if(VERBOSE)
    fprintf(stderr,"%25s: (string) = %s (%s)\n",name,
	    (found ? value : "not found"),
	    (Default != NULL ?  Default : "no default")); 

  return(found);
}

int input_boolean(name,value,interpret,m)  /* supports name=on/off too */
     char *name;
     int *value;
     char *interpret;
     int m;     
{ char *sptr;
  struct arglist *alptr; 
  int h, hno, hyes, found;
  char line[MAXLINE], *str, *noname;

  int essential;
  double Default,minvalue,maxvalue;

  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_boolean: searching for '%s' with default/range '%s'\n",
	    name,(interpret == NULL) ? "**EMPTY**" : interpret);
 
 
  interpret_control_string(interpret,&essential,&Default,&minvalue,&maxvalue);
  
  *value = (int)(Default);
 
  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;

      str= ARGBUF + alptr->argval_offset;
      found=1;
      break;
    } 
 
  if(!found)
    {if(m==0)
      if(VERBOSE)
	if (Default != STRANGE_NUM)
	  fprintf(stderr,"%25s: (boolean int) = not found (%d) \n",name,(int)(Default)); 
	else
	 { fprintf(stderr,"%25s: (boolean int) = not found (no default) \n",name); 
	   if(BEGINNER)
	     { fprintf(stderr,"\t\t Previously set value gives ...");
	       fprintf(stderr,"%d\n",*value);
	     }
	 }
	 
      return(0);
    }
 
  if((strstr(str,"on")!=NULL) || (strstr(str,"ON")!=NULL))
    *value=1;
  else if ((strstr(str,"off") != NULL) || (strstr(str,"OFF")!=NULL))
    *value=0;
  else /* assume some numerical value */
    *value=atoi(str);

  if(m==0)
  if(VERBOSE)
    fprintf(stderr,"%25s: (boolean int) = %d \n",name,*value); 
  
  return(found);
}

int input_float(name,value,interpret,m)
     char *name;
     float *value;
     char *interpret;
     int m;

{ char *sptr;
  struct arglist *alptr;
  
  int h, hno, hyes, found;
  char line[MAXLINE], *str, *noname;
  int exists,essential;
  double Default,minvalue,maxvalue;

  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_float: searching for '%s' with default/range '%s'\n",
	    name,(interpret == NULL) ? "**EMPTY**" : interpret);
 
 
  exists=interpret_control_string(interpret,&essential,&Default,&minvalue,&maxvalue);
 
  *value = (float) Default;

  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;
      str= ARGBUF + alptr->argval_offset;

      sscanf(str,"%f",value);
      found=1;
      break;
    } 
 
  if(essential && !found)
    { fprintf(stderr,"There MUST be an entry for the parameter %s\n",name);
      exit(12);
    }

  if((minvalue!=STRANGE_NUM) && (*value < (float) minvalue))
    *value = (float) minvalue;
  if((maxvalue!=STRANGE_NUM) && (*value > (float) maxvalue))
    *value = (float) maxvalue;

  if(m==0)
  if(VERBOSE)
   { if (found)
       fprintf(stderr,"%25s: (float) = %f \n",name,*value); 
     else
       if (Default != STRANGE_NUM)
	  fprintf(stderr,"%25s: (float) = not found (%f) \n",name,Default); 
       else
	 { fprintf(stderr,"%25s: (float) = not found (no default) \n",name); 
	   if(BEGINNER)
	     { fprintf(stderr,"\t\t Previously set value gives ...");
	       fprintf(stderr,"%g\n",*value);
	     }
	 }
   }
  return(found);
}
  
int input_double(name,value,interpret,m)
     char *name;
     double *value;
     char *interpret;
     int m;

{ char *sptr;
  struct arglist *alptr;
  
  int h, hno, hyes, found;
  char line[MAXLINE], *str, *noname;

  int exists,essential;
  double Default,minvalue,maxvalue;


  if(m==0)
  if(DESCRIBE)
   fprintf(stderr,"input_double: searching for '%s' with default/range '%s'\n",
	   name,(interpret == NULL) ? "**EMPTY**" : interpret);
 
 
  exists=interpret_control_string(interpret,&essential,&Default,&minvalue,&maxvalue);
 

  *value = Default;

  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;
      str= ARGBUF + alptr->argval_offset;
      sscanf(str,"%lf",value);
      found=1;
      break;
    } 
 
  if(essential && !found)
    { fprintf(stderr,"There MUST be an entry for the parameter %s\n",name);
      exit(12);
    }
  if((minvalue!=STRANGE_NUM) && (*value <  minvalue))
    *value =  minvalue;
  if((maxvalue!=STRANGE_NUM) && (*value >  maxvalue))
    *value =  maxvalue;

  if(m==0)
  if(VERBOSE)
   { if (found)
       fprintf(stderr,"%25s: (double) = %g \n",name,*value); 
     else
       if (Default != STRANGE_NUM)
	  fprintf(stderr,"%25s: (double) = not found (%g) \n",name,Default); 
       else
	  { fprintf(stderr,"%25s: (double) = not found (no default) \n",name); 
	    if(BEGINNER)
	       { fprintf(stderr,"\t\t Previously set value gives ...");
		 fprintf(stderr,"%g\n",*value);
	       }
	  }
   }
  

  return(found);
}


int input_int_vector(char *name, int number,int *value,int m)
{ 
  char *sptr;
  struct arglist *alptr;
  char control_string[500];
 
  int h,i, hno, hyes, found;
  char line[MAXLINE], *str, *noname;

  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_int_vector: searching for %s (%d times)\n",name,number);

  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;
      str= ARGBUF + alptr->argval_offset;
      found=1;
      break;
    } 
  /* now interpret vector */
  
  if(!found) return(0);

  for(h=0;h<number;h++)
    { sprintf(control_string,"");
      for(i=0;i<h;i++)
	strcat(control_string,"%*f,");
      strcat(control_string,"%d");
      sscanf(str,control_string,&(value[h]));
    }

  if(m==0)
  if(VERBOSE)
   fprintf(stderr,"%25s: (vector) = %s\n",name,str); 

  return(found);
}



int input_char_vector(name,number,value,m)
     char *name;
     int number;
     char *value; /* comma-separated list of ints */
     int m;

{ char *sptr;
  struct arglist *alptr;
  char control_string[500];
 
  int h,i, hno, hyes, found;
  char line[MAXLINE], *str, *noname;

  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_char_vector: searching for %s (%d times)\n",name,number);

  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;
      str= ARGBUF + alptr->argval_offset;
      found=1;
      break;
    } 
  /* now interpret vector */
  
  if(!found) return(0);

  for(h=0;h<number;h++)
    { sprintf(control_string,"");
      for(i=0;i<h;i++)
	strcat(control_string,"%*c,");
      strcat(control_string,"%c");
      sscanf(str,control_string,&(value[h]));
    }

  if(m==0)
  if(VERBOSE)
   fprintf(stderr,"%25s: (vector) = %s\n",name,str); 

  return(found);
}

int input_float_vector(name,number,value,m)
     char *name;
     int number;
     float *value; /* comma-separated list of floats */
     int m;

{ char *sptr;
  struct arglist *alptr;
  char control_string[500];
 
  int h,i, hno, hyes, found;
  char line[MAXLINE], *str, *noname;

  if(0==number)
      return(0);

  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_float_vector: searching for %s (%d times)\n",name,number);

  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;
      str= ARGBUF + alptr->argval_offset;
      found=1;
      break;
    } 
  /* now interpret vector */
  
  if(!found) return(0);

  for(h=0;h<number;h++)
    { sprintf(control_string,"");
      for(i=0;i<h;i++)
	strcat(control_string,"%*f,");
      strcat(control_string,"%f");
      sscanf(str,control_string,&(value[h]));
    }

  if(m==0)
  if(VERBOSE)
   fprintf(stderr,"%25s: (float vector) = %s\n",name,str); 

  return(found);
}

int input_double_vector(name,number,value,m)
     char *name;
     int number;
     double *value; /* comma-separated list of floats */
     int m;

{ char *sptr;
  struct arglist *alptr;
  char control_string[500];
 
  int h,i, hno, hyes, found;
  char line[MAXLINE], *str, *noname;
 
  if(m==0)
  if(DESCRIBE)
    fprintf(stderr,"input_double_vector: searching for %s (%d times)\n",name,number);

  h=compute_parameter_hash_table(name);
  found=0;

  /* search list backwards, stopping at first find */
  for(alptr= ARGLIST +(NLIST-1); alptr >= ARGHEAD; alptr--)
    { if(alptr->hash != h)
	continue;
      if(strcmp(ARGBUF+alptr->argname_offset,name) != 0)
	continue;
      str= ARGBUF + alptr->argval_offset;
      found=1;
      break;
    } 

  if(!found) return(0);

 /* now interpret vector */
  
  for(h=0;h<number;h++)
    { sprintf(control_string,"");
      for(i=0;i<h;i++)
	strcat(control_string,"%*f,");
      strcat(control_string,"%lf");
      sscanf(str,control_string,&(value[h]));
    }

  if(m==0)
  if(VERBOSE)
   fprintf(stderr,"%25s: (double vector) = %s\n",name,str); 

  return(found);
}

/* =================================================== */
/* This is needed to be fixed on Linux machine
   The function strtok does not work on linux machine
*/

int interpret_control_string(interpret,essential,Default,minvalue,maxvalue)
     char *interpret;
     int *essential;
     double *Default,*minvalue,*maxvalue;

{ char *substring;

  *Default=*maxvalue=*minvalue=STRANGE_NUM;
  *essential=0;
   
  if (strstr(interpret,"essential")!=NULL)
   { *essential=1; /* no default possible, must read a value */
     return(0);
   }
  
  if (strstr(interpret,"nodefault")==NULL) 
   { if((strstr(interpret,"on")!=NULL) || (strstr(interpret,"ON")!=NULL))
       *Default = 1.0;
     else 
       if ((strstr(interpret,"off") != NULL) || (strstr(interpret,"OFF")!=NULL))
	 *Default = 0.0; 
       else
         sscanf(interpret,"%lf",Default);  /* read number as a default value */ 
   }
  
  if ((substring=strstr(interpret,",")) == NULL) /* minvalue */
    { /* no minimum, no maximum */
      return(1);
    }

  if (strstr(substring,"nomin")==NULL)
    sscanf(substring,"%lf",minvalue);
      
  if ((substring=strstr(substring,",")) == NULL) /* maxvalue */
    { /* no maximum */
/*       if (DESCRIBE) */
/* 	fprintf(stderr,"minimum but no maximum\n"); */
      return(2);
    }

  if (strstr(substring,"nomax")==NULL)
    sscanf(substring,"%lf",maxvalue);


  return(0);
 
}
