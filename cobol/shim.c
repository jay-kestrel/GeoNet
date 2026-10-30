/* GeoNet COBOL kernel C shim.
 *
 * Owns the GnuCOBOL runtime startup (one-time cob_init) and the scratch
 * allocation for the per-iteration flux/flag buffers, then forwards to the
 * cobc-generated COBOL program. Compiled together with the cobc-generated C
 * into lib/pygeonet_cobol.so (see Makefile). Direct ctypes into a module
 * without cob_init is not safe - the startup here is load-bearing.
 */
#include "libcob.h"
#include <stdlib.h>

extern int nonlinear_filter(int *rows, int *cols, float *kappa, float *gamma,
                            float *step1, float *step2, int *option,
                            int *niter, int *npix, float *img, float *out,
                            float *nsbuf, float *ewbuf,
                            signed char *snan, signed char *enan);

static int cobol_ready = 0;

/* Called from ctypes with pointers into numpy buffers. */
int ge_anisodiff_cobol(float *img, int rows, int cols, float kappa,
                       float gamma, float step1, float step2, int option,
                       int niter, float *out) {
    if (!cobol_ready) {
        cob_init(0, NULL);
        cobol_ready = 1;
    }
    if (rows <= 0 || cols <= 0 ||
        (long long)rows * (long long)cols > 67108864LL) {
        return 3;
    }
    int npix = rows * cols;
    float *nsbuf = (float *)malloc((size_t)npix * sizeof(float));
    float *ewbuf = (float *)malloc((size_t)npix * sizeof(float));
    signed char *snan = (signed char *)malloc((size_t)npix);
    signed char *enan = (signed char *)malloc((size_t)npix);
    int rc = 1;
    if (nsbuf && ewbuf && snan && enan) {
        rc = nonlinear_filter(&rows, &cols, &kappa, &gamma, &step1, &step2,
                              &option, &niter, &npix, img, out,
                              nsbuf, ewbuf, snan, enan);
    }
    free(nsbuf);
    free(ewbuf);
    free(snan);
    free(enan);
    return rc;
}
