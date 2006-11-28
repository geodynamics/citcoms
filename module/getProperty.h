/*
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/


/* template */


int getTYPEProperty(PyObject* properties, char* attribute, CTYPE *value, FILE* fp)
{
    PyObject *prop;

    if (!(prop = PyObject_GetAttrString(properties, attribute)))
        return -1;

    if (!(PyTYPE_Check(prop))) {
        PyErr_Format(PyExc_ValueError, "'%s': " MESSAGE, attribute);
        return -1;
    }

    if ((CTYPE)-1 == (*value = PyTYPE_AsCTYPE(prop)) &&
        PyErr_Occurred()) {
        return -1;
    }

    if (fp)
        fprintf(fp, "%s=" FORMAT "\n", attribute, *value);

    return 0;
}


int getTYPEVectorProperty(PyObject* properties, char* attribute,
                          CTYPE* vector, int len, FILE* fp)
{
    PyObject *prop, *item;
    Py_ssize_t n;
    int i;
    
    if (!(prop = PyObject_GetAttrString(properties, attribute)))
        return -1;

    if (-1 == (n = PySequence_Size(prop)))
        return -1;
    
    /* is it of length len? */
    if (n < (Py_ssize_t)len) {
        PyErr_Format(PyExc_ValueError,
                     "'%s': too few elements (expected %d, found %d)",
                     attribute, len, n);
	return -1;
    } else if (n > len) {
	if (fp)
            fprintf(fp, "# WARNING: length of '%s' > %d\n", attribute, len);
    }
    
    if (fp)
        fprintf(fp, "%s=", attribute);

    for (i = 0; i < len; i++) {
	item = PySequence_GetItem(prop, i);
	if (!item) {
            PyErr_Format(PyExc_IndexError, "can't get %s[%d]", attribute, i);
	    return -1;
	}

        if (!(PyTYPE_Check(item))) {
            PyErr_Format(PyExc_ValueError, "'%s': " MESSAGE, attribute);
            return -1;
        }

        if ((CTYPE)-1 == (vector[i] = PyTYPE_AsCTYPE(item)) &&
            PyErr_Occurred()) {
            return -1;
        }
        
	if (fp)
            fprintf(fp, "%s" FORMAT, (i ? "," : ""), vector[i]);
    }

    if (fp)
        fprintf(fp, "\n");

    return 0;
}


/* $Id: setProperties.cc 4642 2006-09-28 14:30:32Z luis $ */

/* End of file */
