"""Frozen reference oracle for the COBOL nonlinear-filter port (spec D5).

The ``anisodiff`` below is copied VERBATIM from pygeonet_nonlinear_filter.py
at commit 5e2e879 (master head when the COBOL port landed), minus the
PYGEONET_COBOL dispatch added by the port. It is a test oracle only: the
production module keeps the live implementation, and this copy pins the
parity baseline so future edits to the NumPy path cannot silently move the
COBOL gate (same convention as pygeoflood's frozen anisodiff reference).
"""
import numpy as np


def anisodiff(img, niter, kappa, gamma, step=(1., 1.), option=2):
    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
    step1 = step[0]
    step2 = step[1]
    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        if option == 2:
            #gS = gs_diff(deltaS,kappa,step1)
            #gE = ge_diff(deltaE,kappa,step2)
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
        elif option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't ask questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
        # update the image
        mNS = np.isnan(NS)
        mEW = np.isnan(EW)
        NS[mNS] = 0
        EW[mEW] = 0
        NS += EW
        mNS &= mEW
        NS[mNS] = np.nan
        imgout += gamma*NS
    return imgout
