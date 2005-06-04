# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

PROJECT = CitcomS

# directory structure

DIR_LIB = lib
DIR_DRIVERS = drivers
DIR_MODULE = module
DIR_PYRE = pyre
DIR_ETC = etc
DIR_TESTS = tests
DIR_EXAMPLES = examples


BUILD_DIRS = \
    $(DIR_LIB) \
    $(DIR_DRIVERS) \
    $(DIR_MODULE) \
    $(DIR_PYRE) \
    $(DIR_ETC) \

OTHER_DIRS = \
    $(DIR_TESTS) \
    $(DIR_EXAMPLES)

RECURSE_DIRS = $(BUILD_DIRS)

# targets

all: update

update: $(BUILD_DIRS)

release: tidy
	cvs release .

test: update
	(cd $(DIR_TESTS); $(MM) test)

.PHONY: $(DIR_LIB)
$(DIR_LIB):
	(cd $(DIR_LIB); $(MM))


.PHONY: $(DIR_DRIVERS)
$(DIR_DRIVERS):
	(cd $(DIR_DRIVERS); $(MM))


.PHONY: $(DIR_MODULE)
$(DIR_MODULE):
	(cd $(DIR_MODULE); $(MM))


.PHONY: $(DIR_PYRE)
$(DIR_PYRE):
	(cd $(DIR_PYRE); $(MM))


.PHONY: $(DIR_ETC)
$(DIR_ETC):
	(cd $(DIR_ETC); $(MM))


.PHONY: $(DIR_TESTS)
$(DIR_TESTS):
	(cd $(DIR_TESTS); $(MM))


.PHONY: $(DIR_EXAMPLES)
$(DIR_EXAMPLES):
	(cd $(DIR_EXAMPLES); $(MM))


tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse
	$(RM) $(RMFLAGS) $(BLD_TMPDIR)/$(PROJECT) $(BLD_LIBDIR)/$(PROJECT)

distclean::
	BLD_ACTION="distclean" $(MM) recurse


# version
# $Id: Make.mm,v 1.4 2005/06/03 21:51:39 leif Exp $

#
# End of file
