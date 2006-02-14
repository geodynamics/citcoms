## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:49 leif Exp $

subpackage = journal
include $(top_srcdir)/packages/subpackage.am

SUBDIRS = libjournal journalmodule tests

# applications
dist_bin_SCRIPTS = applications/journald.py

# etc
dist_subpkgsysconf_DATA = etc/__vault__.odb etc/console.odb etc/file.odb etc/remote.odb

# journal
nobase_pkgpyexec_PYTHON = \
	journal/components/ChannelFacility.py \
	journal/components/Channel.py \
	journal/components/Console.py \
	journal/components/DeviceFacility.py \
	journal/components/Device.py \
	journal/components/File.py \
	journal/components/__init__.py \
	journal/components/JournalFacility.py \
	journal/components/Journal.py \
	journal/components/NetRenderer.py \
	journal/components/Remote.py \
	journal/components/RendererFacility.py \
	journal/components/Renderer.py \
	journal/devices/Console.py \
	journal/devices/Device.py \
	journal/devices/File.py \
	journal/devices/__init__.py \
	journal/devices/NetRenderer.py \
	journal/devices/Renderer.py \
	journal/devices/TCPDevice.py \
	journal/devices/TextFile.py \
	journal/devices/UDPDevice.py \
	journal/diagnostics/Debug.py \
	journal/diagnostics/Diagnostic.py \
	journal/diagnostics/Entry.py \
	journal/diagnostics/Error.py \
	journal/diagnostics/Firewall.py \
	journal/diagnostics/Index.py \
	journal/diagnostics/Info.py \
	journal/diagnostics/__init__.py \
	journal/diagnostics/ProxyState.py \
	journal/diagnostics/State.py \
	journal/diagnostics/Warning.py \
	journal/__init__.py \
	journal/Journal.py \
	journal/services/Daemon.py \
	journal/services/__init__.py \
	journal/services/JournalService.py \
	journal/services/Pickler.py

## end of Makefile.am
