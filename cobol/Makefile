# COBOL kernel build for GeoNet.
# Toolchain: GnuCOBOL 3.2 (apt gnucobol3 on ubuntu-latest and dev sandboxes).
# Two-stage build per the validated binding recipe: cobc -C translates the
# COBOL kernel to C, then gcc links it with the shim into one shared object
# that ctypes loads (lib/pygeonet_cobol.so, loaded by
# pygeonet_nonlinear_filter._cobol_lib).

COBOL_SRC := nonlinear_filter.cob
GEN_C     := nonlinear_filter.c
GEN_H     := nonlinear_filter.c.h nonlinear_filter.c.l.h
LIB       := ../lib/pygeonet_cobol.so

.PHONY: lib clean

lib: $(LIB)

$(LIB): $(COBOL_SRC) shim.c
	cobc -C -O2 $(COBOL_SRC)
	mkdir -p ../lib
	gcc -shared -fPIC -O2 -o $@ shim.c $(GEN_C) \
		$$(cob-config --cflags) $$(cob-config --libs)

clean:
	rm -f $(GEN_C) $(GEN_H) $(LIB)
	rmdir ../lib 2>/dev/null || true
