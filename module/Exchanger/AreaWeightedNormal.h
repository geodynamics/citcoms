// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_AreaWeightedNormal_h)
#define pyCitcom_AreaWeightedNormal_h

struct All_variables;
template <class T, int N> class Array2D;
class Boundary;
class FineGridMapping;


class AreaWeightedNormal {
    static const int dim_ = 3;
    const int size_;
    const double toleranceOutflow_;
    double* nwght;

public:
    AreaWeightedNormal(const Boundary* boundary,
		       const All_variables* E,
		       const FineGridMapping* fgmapping);
    ~AreaWeightedNormal();

    typedef Array2D<double,dim_> Velo;

    void imposeConstraint(Velo& V) const;

private:
    void computeWeightedNormal(const Boundary* boundary,
			       const All_variables* E,
			       const FineGridMapping* fgmapping);
    double computeOutflow(const Velo& V) const;
    void reduceOutflow(Velo& V, const double outflow) const;

};


#endif

// version
// $Id: AreaWeightedNormal.h,v 1.1 2003/10/20 17:13:08 tan2 Exp $

// End of file

