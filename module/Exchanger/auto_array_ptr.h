// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(auto_array_ptr_h)
#define auto_array_ptr_h

template<class X>
class auto_array_ptr {
    X* p_;

public:
    inline auto_array_ptr(X* p = NULL) throw() : p_(p) {}
    inline auto_array_ptr(auto_array_ptr<X>& ap) throw() : p_(ap.release()) {}
    inline ~auto_array_ptr() {delete [] p_;}

    inline void operator=(auto_array_ptr<X>& rhs) {
	if (this != &rhs) {
	    remove(p_);
	    p_ = rhs.release();
	}
    }

    inline X& operator*() throw() {return *p_;}
    inline X& operator[](int i) throw() {return p_[i];}
    inline X operator[](int i) const throw() {return p_[i];}

    inline X* get() const throw() {return p_;}
    inline X* release() throw() {return reset(NULL);}
    inline X* reset(X* p) throw() {X* tmp = p_; p_ = p; return tmp;}

    static void remove(X*& x) {X* tmp = x; x = NULL; delete [] tmp;}
};

#endif

// version
// $Id: auto_array_ptr.h,v 1.1 2003/10/03 18:25:56 tan2 Exp $

// End of file
