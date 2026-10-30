>>SOURCE FORMAT FREE
*> ---------------------------------------------------------------------------
*> GeoNet COBOL kernel: Perona-Malik anisotropic diffusion (nonlinear filter)
*> ---------------------------------------------------------------------------
*> COBOL port of pygeonet_nonlinear_filter.anisodiff at 5e2e879. The frozen
*> Python oracle in tests/reference_nonlinear_filter.py is the parity
*> baseline; operation order below matches that reference statement by
*> statement so intermediate float32 rounding matches the NumPy chain.
*>
*> Numeric contract (spec art_QEojt4gd, locked decisions 6-7):
*> - COMP-1 (float32) arithmetic; one COBOL statement per NumPy float32 op.
*> - GnuCOBOL COMPUTE does not propagate NaN operands (it coerces them
*>   toward zero - verified by probe), so NaN is tracked with explicit
*>   flags and never enters arithmetic. Values at flagged positions are
*>   placeholders (0.0) and are never used.
*> - RNG stays at the Python layer: the kernel is a pure function of the
*>   buffers it receives (no state, no draws).
*>
*> Buffers: LS-IMG (in, never written), LS-OUT (result), LS-NSBUF/LS-EWBUF
*> (float scratch for the per-iteration flux seeds S/E), LS-SNAN/LS-ENAN
*> (one NaN flag byte per cell). Scratch is allocated by the C shim.
*>
*> Return codes: 0 ok, 2 kappa is zero (reference would emit inf with a
*> NumPy warning; the COBOL backend rejects it - documented divergence).
IDENTIFICATION DIVISION.
PROGRAM-ID. NONLINEAR-FILTER AS "nonlinear_filter".
DATA DIVISION.
WORKING-STORAGE SECTION.
01 WS-NAN-CONST.
   05 WS-NAN-INT  BINARY-LONG VALUE 2143289344.
   *> 0x7FC00000: quiet float32 NaN, synthesized only via this alias
   05 WS-NAN-VAL  REDEFINES WS-NAN-INT COMP-1.
01 WS-IT        BINARY-LONG.
01 WS-I         BINARY-LONG.
01 WS-J         BINARY-LONG.
01 WS-IDX       BINARY-LONG.
01 WS-IDX-N     BINARY-LONG.
01 WS-IDX-W     BINARY-LONG.
01 WS-A         COMP-1.
01 WS-B         COMP-1.
01 WS-DS        COMP-1.
01 WS-DE        COMP-1.
01 WS-R         COMP-1.
01 WS-T         COMP-1.
01 WS-G         COMP-1.
01 WS-S         COMP-1.
01 WS-E         COMP-1.
01 WS-NS        COMP-1.
01 WS-EW        COMP-1.
01 WS-FLUX      COMP-1.
01 WS-TMP       COMP-1.
01 A-NAN        BINARY-CHAR.
01 DS-NAN       BINARY-CHAR.
01 DE-NAN       BINARY-CHAR.
01 S-NAN        BINARY-CHAR.
01 E-NAN        BINARY-CHAR.
01 NS-NAN       BINARY-CHAR.
01 EW-NAN       BINARY-CHAR.
01 FLUX-NAN     BINARY-CHAR.
LINKAGE SECTION.
01 LS-ROWS      BINARY-LONG.
01 LS-COLS      BINARY-LONG.
01 LS-KAPPA     COMP-1.
01 LS-GAMMA     COMP-1.
01 LS-STEP1     COMP-1.
01 LS-STEP2     COMP-1.
01 LS-OPTION    BINARY-LONG.
01 LS-NITER     BINARY-LONG.
01 LS-NPIX      BINARY-LONG.
01 LS-IMG.
   05 LS-IMG-PIX OCCURS 1 TO 67108864 DEPENDING ON LS-NPIX COMP-1.
01 LS-OUT.
   05 LS-OUT-PIX OCCURS 1 TO 67108864 DEPENDING ON LS-NPIX COMP-1.
01 LS-NSBUF.
   05 LS-NS-PIX  OCCURS 1 TO 67108864 DEPENDING ON LS-NPIX COMP-1.
01 LS-EWBUF.
   05 LS-EW-PIX  OCCURS 1 TO 67108864 DEPENDING ON LS-NPIX COMP-1.
01 LS-SNAN.
   05 LS-SNAN-B  OCCURS 1 TO 67108864 DEPENDING ON LS-NPIX BINARY-CHAR.
01 LS-ENAN.
   05 LS-ENAN-B  OCCURS 1 TO 67108864 DEPENDING ON LS-NPIX BINARY-CHAR.
PROCEDURE DIVISION USING LS-ROWS LS-COLS LS-KAPPA LS-GAMMA LS-STEP1
                         LS-STEP2 LS-OPTION LS-NITER LS-NPIX
                         LS-IMG LS-OUT LS-NSBUF LS-EWBUF LS-SNAN LS-ENAN.
    IF LS-KAPPA = 0
       MOVE 2 TO RETURN-CODE
       GOBACK
    END-IF
    IF LS-NITER < 1
       MOVE LS-IMG TO LS-OUT
       MOVE 0 TO RETURN-CODE
       GOBACK
    END-IF
    PERFORM VARYING WS-IT FROM 1 BY 1 UNTIL WS-IT > LS-NITER
        *> -------------------------------------------------------------
        *> Pass 1: deltas, conductances, flux seeds S -> NSBUF, E -> EWBUF
        *> (reference: deltaS/deltaE via np.diff, gS/gE per option, E/S)
        *> -------------------------------------------------------------
        PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > LS-ROWS
            PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > LS-COLS
                COMPUTE WS-IDX = (WS-I - 1) * LS-COLS + WS-J
                MOVE LS-IMG-PIX(WS-IDX) TO WS-A
                IF WS-A NOT = WS-A
                   MOVE 1 TO A-NAN
                   MOVE 0.0 TO WS-A
                ELSE
                   MOVE 0 TO A-NAN
                END-IF
                *> south delta: img[i+1,j] - img[i,j], zero on the last row
                IF WS-I < LS-ROWS
                    COMPUTE WS-IDX-N = WS-IDX + LS-COLS
                    MOVE LS-IMG-PIX(WS-IDX-N) TO WS-B
                    IF WS-B NOT = WS-B
                       MOVE 1 TO DS-NAN
                       MOVE 0.0 TO WS-B
                    ELSE
                       MOVE 0 TO DS-NAN
                    END-IF
                    IF A-NAN = 1
                       MOVE 1 TO DS-NAN
                    END-IF
                    COMPUTE WS-DS = WS-B - WS-A
                ELSE
                    MOVE 0.0 TO WS-DS
                    MOVE 0 TO DS-NAN
                END-IF
                *> east delta: img[i,j+1] - img[i,j], zero on the last column
                IF WS-J < LS-COLS
                    COMPUTE WS-IDX-W = WS-IDX + 1
                    MOVE LS-IMG-PIX(WS-IDX-W) TO WS-B
                    IF WS-B NOT = WS-B
                       MOVE 1 TO DE-NAN
                       MOVE 0.0 TO WS-B
                    ELSE
                       MOVE 0 TO DE-NAN
                    END-IF
                    IF A-NAN = 1
                       MOVE 1 TO DE-NAN
                    END-IF
                    COMPUTE WS-DE = WS-B - WS-A
                ELSE
                    MOVE 0.0 TO WS-DE
                    MOVE 0 TO DE-NAN
                END-IF
                *> PeronaMalik1: g = exp(-(d/k)**2)/step (option = 1)
                *> PeronaMalik2: g = 1/(1+(d/k)**2)/step (option = 2)
                *> other options: the reference leaves g at its initial 1.0
                IF DS-NAN = 1
                   MOVE 0.0 TO WS-S
                   MOVE 1 TO S-NAN
                ELSE
                    IF LS-OPTION = 1
                       COMPUTE WS-R = WS-DS / LS-KAPPA
                       COMPUTE WS-T = WS-R * WS-R
                       *> GnuCOBOL's EXP evaluates huge-magnitude arguments in
                       *> arbitrary-precision decimal and aborts on the
                       *> allocation; float32 exp(-x) is 0 for x >= 104, so
                       *> clamp there - matches the NumPy underflow to zero.
                       IF WS-T >= 104.0
                          MOVE 0.0 TO WS-G
                       ELSE
                          COMPUTE WS-G = FUNCTION EXP(0.0 - WS-T)
                       END-IF
                       COMPUTE WS-G = WS-G / LS-STEP1
                    ELSE
                       IF LS-OPTION = 2
                          COMPUTE WS-R = WS-DS / LS-KAPPA
                          COMPUTE WS-T = WS-R * WS-R
                          COMPUTE WS-T = 1.0 + WS-T
                          COMPUTE WS-G = 1.0 / WS-T
                          COMPUTE WS-G = WS-G / LS-STEP1
                       ELSE
                          MOVE 1.0 TO WS-G
                       END-IF
                    END-IF
                    COMPUTE WS-S = WS-G * WS-DS
                    MOVE 0 TO S-NAN
                END-IF
                IF DE-NAN = 1
                   MOVE 0.0 TO WS-E
                   MOVE 1 TO E-NAN
                ELSE
                    IF LS-OPTION = 1
                       COMPUTE WS-R = WS-DE / LS-KAPPA
                       COMPUTE WS-T = WS-R * WS-R
                       IF WS-T >= 104.0
                          MOVE 0.0 TO WS-G
                       ELSE
                          COMPUTE WS-G = FUNCTION EXP(0.0 - WS-T)
                       END-IF
                       COMPUTE WS-G = WS-G / LS-STEP2
                    ELSE
                       IF LS-OPTION = 2
                          COMPUTE WS-R = WS-DE / LS-KAPPA
                          COMPUTE WS-T = WS-R * WS-R
                          COMPUTE WS-T = 1.0 + WS-T
                          COMPUTE WS-G = 1.0 / WS-T
                          COMPUTE WS-G = WS-G / LS-STEP2
                       ELSE
                          MOVE 1.0 TO WS-G
                       END-IF
                    END-IF
                    COMPUTE WS-E = WS-G * WS-DE
                    MOVE 0 TO E-NAN
                END-IF
                MOVE WS-S TO LS-NS-PIX(WS-IDX)
                MOVE S-NAN TO LS-SNAN-B(WS-IDX)
                MOVE WS-E TO LS-EW-PIX(WS-IDX)
                MOVE E-NAN TO LS-ENAN-B(WS-IDX)
            END-PERFORM
        END-PERFORM
        *> -------------------------------------------------------------
        *> Pass 2: flux divergence + NaN mask + image update
        *> (reference: NS = S - shifted S, EW = E - shifted E, mask, out)
        *> -------------------------------------------------------------
        PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > LS-ROWS
            PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > LS-COLS
                COMPUTE WS-IDX = (WS-I - 1) * LS-COLS + WS-J
                *> NS[i,j] = S[i,j] - S[i-1,j]; row 1 has no north neighbor.
                *> Scratch is read-only here, so ordering is free.
                IF WS-I > 1
                    COMPUTE WS-IDX-N = WS-IDX - LS-COLS
                    COMPUTE WS-NS = LS-NS-PIX(WS-IDX) - LS-NS-PIX(WS-IDX-N)
                    IF LS-SNAN-B(WS-IDX) = 1 OR LS-SNAN-B(WS-IDX-N) = 1
                       MOVE 1 TO NS-NAN
                    ELSE
                       MOVE 0 TO NS-NAN
                    END-IF
                ELSE
                    MOVE LS-NS-PIX(WS-IDX) TO WS-NS
                    MOVE LS-SNAN-B(WS-IDX) TO NS-NAN
                END-IF
                *> EW[i,j] = E[i,j] - E[i,j-1]; column 1 has no west neighbor
                IF WS-J > 1
                    COMPUTE WS-IDX-W = WS-IDX - 1
                    COMPUTE WS-EW = LS-EW-PIX(WS-IDX) - LS-EW-PIX(WS-IDX-W)
                    IF LS-ENAN-B(WS-IDX) = 1 OR LS-ENAN-B(WS-IDX-W) = 1
                       MOVE 1 TO EW-NAN
                    ELSE
                       MOVE 0 TO EW-NAN
                    END-IF
                ELSE
                    MOVE LS-EW-PIX(WS-IDX) TO WS-EW
                    MOVE LS-ENAN-B(WS-IDX) TO EW-NAN
                END-IF
                *> reference mask order: NaN -> 0, then both-NaN -> NaN
                IF NS-NAN = 1
                   MOVE 0.0 TO WS-NS
                END-IF
                IF EW-NAN = 1
                   MOVE 0.0 TO WS-EW
                END-IF
                COMPUTE WS-FLUX = WS-NS + WS-EW
                IF NS-NAN = 1 AND EW-NAN = 1
                   MOVE WS-NAN-VAL TO WS-FLUX
                   MOVE 1 TO FLUX-NAN
                ELSE
                   MOVE 0 TO FLUX-NAN
                END-IF
                *> imgout += gamma * flux; a NaN image cell stays NaN
                IF FLUX-NAN = 1
                   MOVE WS-NAN-VAL TO LS-OUT-PIX(WS-IDX)
                ELSE
                    MOVE LS-IMG-PIX(WS-IDX) TO WS-A
                    IF WS-A NOT = WS-A
                       MOVE WS-NAN-VAL TO LS-OUT-PIX(WS-IDX)
                    ELSE
                       COMPUTE WS-TMP = LS-GAMMA * WS-FLUX
                       COMPUTE LS-OUT-PIX(WS-IDX) = WS-A + WS-TMP
                    END-IF
                END-IF
            END-PERFORM
        END-PERFORM
        *> next iteration's deltas must see the updated image
        MOVE LS-OUT TO LS-IMG
    END-PERFORM
    MOVE 0 TO RETURN-CODE
    GOBACK.
