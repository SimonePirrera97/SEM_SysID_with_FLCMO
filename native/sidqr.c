#define _POSIX_C_SOURCE 199309L
/* ============================================================================
 *  sidqr.c -- implementation.  Portable C99, no external dependencies.
 *
 *  The factorisation is organised around three structural facts, derived in
 *  Section 3 of the accompanying report:
 *
 *  (F1) at step k only rows k .. bbot(k) of the working matrix can be
 *       nonzero; their number is the frontal height h(k);
 *  (F2) rows 1..ntheta of X are dense, so after the first reflection the
 *       frontal window is dense in every remaining column and R is
 *       structurally full: there is no sparsity left to exploit *inside*
 *       the window;
 *  (F3) therefore the working storage is a dense strip of h(k) rows that
 *       slides down as k advances -- finished rows of R leave at the top,
 *       original data enters at the bottom.
 *
 *  On top of that the sweep is blocked (compact WY) and the trailing update
 *  is register-tiled over SIDQR_NCB columns, which is what makes it compute
 *  bound instead of memory-bandwidth bound.
 * ==========================================================================*/
#include "sidqr.h"

#ifndef SIDQR_NCB
#define SIDQR_NCB 4     /* trailing columns handled per register tile */
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef SIDQR_USE_MWBLAS
#include "blas.h"
#endif

/* ------------------------------------------------------------------ utils */
static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static unsigned long long rng_s;
static double urand(void)                    /* xorshift64* -> (0,1) */
{
    rng_s ^= rng_s >> 12; rng_s ^= rng_s << 25; rng_s ^= rng_s >> 27;
    unsigned long long z = rng_s * 2685821657736338717ULL;
    return ((double)(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static double nrand(void)                    /* Box-Muller */
{
    double u1 = urand(), u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

int sidqr_setdims(sidqr_dims *d, int N, int n, int p, int q, int ntheta)
{
    if (!d || N <= 0 || n < 0 || p <= 0 || q < 0 || ntheta < 0 || N <= n)
        return -1;
    d->N = N; d->n = n; d->p = p; d->q = q; d->ntheta = ntheta;
    d->gamma = p + q;
    d->beta  = (n + 1) * d->gamma;
    d->m     = p * (N - n);
    d->nu    = ntheta + d->gamma * N;
    return 0;
}

void sidqr_gen(const sidqr_dims *d, double *Th, double *Bd, unsigned seed)
{
    int j, i, l, ii;
    rng_s = 88172645463325252ULL ^ (unsigned long long)seed * 2654435761ULL;
    for (j = 0; j < d->m; ++j)
        for (i = 0; i < d->ntheta; ++i)
            Th[i + (size_t)d->ntheta * j] = 0.30 * nrand();

    for (j = 0; j < d->m; ++j)
        for (i = 0; i < d->beta; ++i)
            Bd[i + (size_t)d->beta * j] = 0.30 * nrand();

    /* dh_l/dy_t = -I_p  :  last p rows of the band of each column block.
       This is what makes J full row rank for every xi (Lemma, Sec. 4).  */
    for (l = 0; l < d->N - d->n; ++l)
        for (ii = 0; ii < d->p; ++ii) {
            j = d->p * l + ii;
            for (i = d->beta - d->p; i < d->beta; ++i)
                Bd[i + (size_t)d->beta * j] = 0.0;
            Bd[(d->beta - d->p + ii) + (size_t)d->beta * j] = -1.0;
        }
}

/* ------------------------------------------------------------------------ */
/*  Householder generator (LAPACK dlarfg convention).                        */
/*  On exit x[0] = R(k,k), x[1..len-1] = v(2:len), v(1) = 1 implicitly,      */
/*  H = I - tau*v*v^T.  The sign is chosen so that no cancellation occurs.   */
/* ------------------------------------------------------------------------ */
static double housegen(double *x, int len, double *flops)
{
    double alpha, xnorm2 = 0.0, xnorm, bta, tau, scal;
    int i;
    if (len <= 1) return 0.0;
    alpha = x[0];
    for (i = 1; i < len; ++i) xnorm2 += x[i] * x[i];
    *flops += 2.0 * (len - 1);
    if (xnorm2 == 0.0) return 0.0;
    xnorm = sqrt(xnorm2);
    bta   = (alpha >= 0.0) ? -sqrt(alpha * alpha + xnorm2)
                           :  sqrt(alpha * alpha + xnorm2);
    tau   = (bta - alpha) / bta;
    scal  = 1.0 / (alpha - bta);
    for (i = 1; i < len; ++i) x[i] *= scal;
    *flops += (double)(len - 1) + 8.0;
    x[0] = bta;
    (void)xnorm;
    return tau;
}

/* =========================================================================
 *  Blocked structured Householder QR with a sliding dense frontal strip.
 * ======================================================================= */
#define SIDQR_RCOL(J) (Rld ? (Rp + (size_t)Rld * (size_t)(J)) \
                            : (Rp + ((size_t)(J) * ((size_t)(J) + 1)) / 2))

static int hh_core(const sidqr_dims *d, const double *Th, const double *Bd,
                   int blk, double *Rp, int Rld, int Rsym,
                   double *Y, int nrhs, sidqr_stats *st, int backend)
{
    const int m = d->m, p = d->p, gam = d->gamma, bet = d->beta, nth = d->ntheta;
    int Hs = 0, ldS, head = 0, k0, kend, bcur, H, Hnext, jj, t, i, j, r, inj = 0;
    size_t peak = 0;
    double *S = NULL, *T = NULL, *tau = NULL, *w = NULL, *w2 = NULL;
#ifdef SIDQR_USE_MWBLAS
    double *VB = NULL, *WB = NULL, *W2B = NULL;
#endif
    int    *Ht = NULL;
    double fl = 0.0, t0;

    if (blk <= 0) blk = p;
    if (blk % p) blk = ((blk + p - 1) / p) * p;      /* round up to mult. of p */

    /* peak frontal height and peak live storage, known a priori */
    for (k0 = 1; k0 <= m; k0 += blk) {
        kend = k0 + blk - 1; if (kend > m) kend = m;
        H = sidqr_bbot(d, kend) - k0 + 1;
        if (H > Hs) Hs = H;
        if ((size_t)H * (size_t)(m - k0 + 1) > peak)
            peak = (size_t)H * (size_t)(m - k0 + 1);
    }

    ldS = Hs + (backend == 2 ? 16 * blk : 0);
    S   = (double *)calloc((size_t)ldS * (size_t)m, sizeof(double));
    T   = (double *)calloc((size_t)blk * (size_t)blk, sizeof(double));
    tau = (double *)calloc((size_t)blk, sizeof(double));
    w   = (double *)calloc((size_t)blk * SIDQR_NCB, sizeof(double));
    w2  = (double *)calloc((size_t)blk * SIDQR_NCB, sizeof(double));
    Ht  = (int    *)calloc((size_t)blk, sizeof(int));
#ifdef SIDQR_USE_MWBLAS
    if (backend) {
        VB  = (double *)calloc((size_t)Hs * (size_t)blk, sizeof(double));
        WB  = (double *)malloc((size_t)blk * (size_t)m * sizeof(double));
        W2B = (double *)malloc((size_t)blk * (size_t)m * sizeof(double));
    }
#else
    (void)backend;
#endif
    if (!S || !T || !tau || !w || !w2 || !Ht) { free(S); free(T); free(tau);
                                    free(w); free(w2); free(Ht); return -2; }
#ifdef SIDQR_USE_MWBLAS
    if (backend && (!VB || !WB || !W2B)) {
        free(S); free(T); free(tau); free(w); free(w2); free(Ht);
        free(VB); free(WB); free(W2B); return -2;
    }
#endif

    t0 = wtime();
    for (k0 = 1; k0 <= m; k0 += blk) {
        kend = k0 + blk - 1; if (kend > m) kend = m;
        bcur = kend - k0 + 1;
        H    = sidqr_bbot(d, kend) - k0 + 1;

        /* ---- 1. inject the original entries of the rows that just entered */
        for (r = inj + 1; r <= k0 + H - 1; ++r) {
            int sr = r - k0;                       /* strip row */
            if (r <= nth) {
                const double *src = Th + (r - 1);
                for (j = 0; j < m; ++j) S[head + sr + (size_t)ldS * j] = src[(size_t)nth * j];
            } else {
                int rr = r - nth;                  /* 1 .. gamma*N */
                int lr = (rr + gam - 1) / gam;
                int lo = lr - d->n, hi = lr, l;
                if (lo < 1) lo = 1;
                if (hi > d->N - d->n) hi = d->N - d->n;
                for (l = lo; l <= hi; ++l) {
                    int off = rr - gam * (l - 1) - 1;
                    if (off < 0 || off >= bet) continue;
                    for (i = 0; i < p; ++i) {
                        j = p * (l - 1) + i;
                        S[head + sr + (size_t)ldS * j] = Bd[off + (size_t)bet * j];
                    }
                }
            }
        }
        if (k0 + H - 1 > inj) inj = k0 + H - 1;

        /* ---- 2. unblocked factorisation of the panel -------------------
           The support of the jj-th reflector stops at the last row that can
           be nonzero in column k0+jj, i.e. at Ht[jj]; operating beyond that
           row would multiply structural zeros.  Without this refinement the
           blocked sweep would perform ~ blk/(2h) extra operations.       */
        for (jj = 0; jj < bcur; ++jj) Ht[jj] = sidqr_bbot(d, k0 + jj) - k0 + 1;
        for (jj = 0; jj < bcur; ++jj) {
            double *x = &S[head + jj + (size_t)ldS * (k0 - 1 + jj)];
            int hj = Ht[jj], len = hj - jj, cc;
            tau[jj] = housegen(x, len, &fl);
            for (cc = jj + 1; cc < bcur; ++cc) {
                double *a = &S[head + (size_t)ldS * (k0 - 1 + cc)];
                double s = a[jj];
                for (i = jj + 1; i < hj; ++i) s += x[i - jj] * a[i];
                s *= tau[jj];
                a[jj] -= s;
                for (i = jj + 1; i < hj; ++i) a[i] -= x[i - jj] * s;
                fl += 4.0 * (double)len;
            }
        }

        /* ---- 3. compact WY factor:  H_1...H_bcur = I - V T V^T --------- */
        for (jj = 0; jj < bcur; ++jj) {
            const double *vj = &S[head + (size_t)ldS * (k0 - 1 + jj)];
            T[jj + (size_t)blk * jj] = tau[jj];
            for (t = 0; t < jj; ++t) {
                const double *vt = &S[head + (size_t)ldS * (k0 - 1 + t)];
                double s = vt[jj];
                for (i = jj + 1; i < Ht[t]; ++i) s += vt[i] * vj[i];
                w[t] = s;
                fl += 2.0 * (double)(Ht[t] - jj);
            }
            for (t = 0; t < jj; ++t) {
                double s = 0.0; int u;
                for (u = t; u < jj; ++u) s += T[t + (size_t)blk * u] * w[u];
                T[t + (size_t)blk * jj] = -tau[jj] * s;
                fl += 2.0 * (double)(jj - t);
            }
        }

        /* ---- 4. store the diagonal block of R -------------------------- */
        if (Rp)
            for (jj = 0; jj < bcur; ++jj) {
                int cj = k0 - 1 + jj;
                double *dst = SIDQR_RCOL(cj) + (k0 - 1);
                const double *src = &S[head + (size_t)ldS * cj];
                for (i = 0; i <= jj; ++i) dst[i] = src[i];
            }

        /* ---- 4b. fused forward substitution  R^T y = rhs ---------------
           Rows k0..kend of R are final at this point and the diagonal block
           D = R(k0:kend,k0:kend) sits in the first bcur rows of the panel
           columns of the strip.  Solving D^T y = s here, and subtracting the
           contribution of those rows from the trailing entries of s inside
           the update loop below, makes the forward substitution free of
           memory traffic: it consumes the rows of R while they are still in
           cache and never reads the stored factor back.                  */
        if (Y) {
            int e;
            for (e = 0; e < nrhs; ++e) {
                double *ye = Y + (size_t)m * e;
                for (t = 0; t < bcur; ++t) {
                    const double *Dt = &S[head + (size_t)ldS * (k0 - 1 + t)];
                    double val = ye[k0 - 1 + t];
                    int u;
                    for (u = 0; u < t; ++u) val -= Dt[u] * ye[k0 - 1 + u];
                    ye[k0 - 1 + t] = val / Dt[t];
                    fl += 2.0 * (double)t + 1.0;
                }
            }
        }

        /* ---- 4c. optional transposed copy of the diagonal block -------- */
        if (Rp && Rld && Rsym)
            for (jj = 0; jj < bcur; ++jj) {
                int cj = k0 - 1 + jj;
                const double *src = &S[head + (size_t)ldS * cj];
                for (i = 0; i <= jj; ++i)
                    Rp[cj + (size_t)Rld * (k0 - 1 + i)] = src[i];
            }

        /* ---- 5. trailing update, R write-out and strip shift, fused ---- */
        if (kend < m) {
            int kend2 = kend + blk; if (kend2 > m) kend2 = m;
            Hnext = sidqr_bbot(d, kend2) - (k0 + bcur) + 1;
        } else Hnext = 0;

        /* Trailing columns are processed in groups of SIDQR_NCB so that
           every element of V loaded from cache is reused NCB times
           (register blocking).  This is what turns the update from a
           bandwidth bound BLAS-2 sweep into a compute bound kernel.      */
        j = kend;
#ifdef SIDQR_USE_MWBLAS
        if (backend && j < m) {
            const char nt = 'N', tr = 'T';
            const double one = 1.0, zero = 0.0, minus_one = -1.0;
            ptrdiff_t ph = H, pb = bcur, pn = m - j;
            ptrdiff_t ldv = Hs, ldt = blk, lds = ldS, ldw = blk;

            /* Explicit V removes the R entries above each reflector and
               pads its unequal support with zeros.  This regular dense
               representation lets optimized dgemm process the full tail. */
            memset(VB, 0, (size_t)Hs * (size_t)bcur * sizeof(double));
            for (t = 0; t < bcur; ++t) {
                const double *src = &S[head + (size_t)ldS * (k0 - 1 + t)];
                double *dst = &VB[(size_t)Hs * t];
                dst[t] = 1.0;
                for (i = t + 1; i < Ht[t]; ++i) dst[i] = src[i];
            }

            /* W=V^T A; W2=T^T W; A=A-V W2. */
            dgemm(&tr, &nt, &pb, &pn, &ph, &one, VB, &ldv,
                  &S[head + (size_t)ldS * j], &lds, &zero, WB, &ldw);
            dgemm(&tr, &nt, &pb, &pn, &pb, &one, T, &ldt,
                  WB, &ldw, &zero, W2B, &ldw);
            dgemm(&nt, &nt, &ph, &pn, &pb, &minus_one, VB, &ldv,
                  W2B, &ldw, &one, &S[head + (size_t)ldS * j], &lds);
            fl += (4.0 * (double)H * bcur + 2.0 * bcur * bcur)
                  * (double)(m - j);

            for (; j < m; ++j) {
                double *a = &S[head + (size_t)ldS * j];
                if (Rp) {
                    double *dst = SIDQR_RCOL(j) + (k0 - 1);
                    for (t = 0; t < bcur; ++t) dst[t] = a[t];
                    if (Rld && Rsym)
                        for (t = 0; t < bcur; ++t)
                            Rp[j + (size_t)Rld * (k0 - 1 + t)] = a[t];
                }
                if (Y) {
                    int e; for (e = 0; e < nrhs; ++e) {
                        double *ye = Y + (size_t)m * e;
                        double acc = 0.0;
                        for (t = 0; t < bcur; ++t)
                            acc += a[t] * ye[k0 - 1 + t];
                        ye[j] -= acc;
                    }
                    fl += 2.0 * (double)bcur * (double)nrhs;
                }
                if (backend == 1) {
                    memmove(a, a + bcur, (size_t)(H - bcur) * sizeof(double));
                    if (Hnext > H - bcur)
                        memset(a + H - bcur, 0,
                               (size_t)(Hnext - (H - bcur)) * sizeof(double));
                } else if (Hnext > H - bcur) {
                    /* Lazy strip: the retained rows already begin at
                       head+bcur.  Only clear the newly entering tail. */
                    memset(a + H, 0,
                           (size_t)(Hnext - (H - bcur)) * sizeof(double));
                }
            }
            if (backend == 2 && kend < m) {
                head += bcur;
                if (head + Hnext > ldS) {
                    for (j = kend; j < m; ++j) {
                        double *col = &S[(size_t)ldS * j];
                        memmove(col, col + head, (size_t)Hnext * sizeof(double));
                    }
                    head = 0;
                }
            }
        }
#endif
        while (j < m) {
            int nc = m - j; if (nc > SIDQR_NCB) nc = SIDQR_NCB;
            if (nc == SIDQR_NCB) {
                double *a0=&S[head+(size_t)ldS*j],     *a1=&S[head+(size_t)ldS*(j+1)];
                double *a2=&S[head+(size_t)ldS*(j+2)], *a3=&S[head+(size_t)ldS*(j+3)];
                for (t = 0; t < bcur; ++t) {          /* W = V^T A        */
                    const double *vt = &S[head+(size_t)ldS*(k0-1+t)];
                    double s0=a0[t],s1=a1[t],s2=a2[t],s3=a3[t]; int ht=Ht[t];
                    for (i = t+1; i < ht; ++i) {
                        double vi = vt[i];
                        s0 += vi*a0[i]; s1 += vi*a1[i];
                        s2 += vi*a2[i]; s3 += vi*a3[i];
                    }
                    w[t]=s0; w[t+blk]=s1; w[t+2*blk]=s2; w[t+3*blk]=s3;
                    fl += 2.0*(double)SIDQR_NCB*(double)(ht-t);
                }
                for (t = 0; t < bcur; ++t) {          /* W = T^T W        */
                    double s0=0,s1=0,s2=0,s3=0; int u;
                    for (u = 0; u <= t; ++u) {
                        double tu = T[u + (size_t)blk*t];
                        s0+=tu*w[u]; s1+=tu*w[u+blk];
                        s2+=tu*w[u+2*blk]; s3+=tu*w[u+3*blk];
                    }
                    w2[t]=s0; w2[t+blk]=s1; w2[t+2*blk]=s2; w2[t+3*blk]=s3;
                    fl += 2.0*(double)SIDQR_NCB*(double)(t+1);
                }
                for (t = 0; t < bcur; ++t) {          /* A = A - V W      */
                    const double *vt = &S[head+(size_t)ldS*(k0-1+t)];
                    double u0=w2[t],u1=w2[t+blk],u2=w2[t+2*blk],u3=w2[t+3*blk];
                    int ht=Ht[t];
                    a0[t]-=u0; a1[t]-=u1; a2[t]-=u2; a3[t]-=u3;
                    for (i = t+1; i < ht; ++i) {
                        double vi = vt[i];
                        a0[i]-=vi*u0; a1[i]-=vi*u1;
                        a2[i]-=vi*u2; a3[i]-=vi*u3;
                    }
                    fl += 2.0*(double)SIDQR_NCB*(double)(ht-t);
                }
                if (Rp) {
                    int c; for (c = 0; c < SIDQR_NCB; ++c) {
                        double *dst = SIDQR_RCOL(j+c) + (k0-1);
                        const double *a = &S[head+(size_t)ldS*(j+c)];
                        for (t = 0; t < bcur; ++t) dst[t] = a[t];
                        if (Rld && Rsym)
                            for (t = 0; t < bcur; ++t)
                                Rp[(j+c) + (size_t)Rld*(k0-1+t)] = a[t];
                    }
                }
                if (Y) {
                    /* four independent reduction chains, one per column of
                       the register tile, so that the fused forward
                       substitution is not latency bound on a length-b
                       serial sum */
                    int e; for (e = 0; e < nrhs; ++e) {
                        double *ye = Y + (size_t)m*e, *yk = ye + (k0-1);
                        double c0=0.0,c1=0.0,c2=0.0,c3=0.0;
                        for (t = 0; t < bcur; ++t) {
                            double yt = yk[t];
                            c0 += a0[t]*yt; c1 += a1[t]*yt;
                            c2 += a2[t]*yt; c3 += a3[t]*yt;
                        }
                        ye[j]-=c0; ye[j+1]-=c1; ye[j+2]-=c2; ye[j+3]-=c3;
                    }
                    fl += 2.0*(double)SIDQR_NCB*(double)bcur*(double)nrhs;
                }
                { int c; for (c = 0; c < SIDQR_NCB; ++c) {
                    double *a = &S[head+(size_t)ldS*(j+c)];
                    for (i = bcur; i < H; ++i) a[i-bcur] = a[i];
                    for (i = H-bcur; i < Hnext; ++i) a[i] = 0.0; } }
            } else {
                int c; for (c = 0; c < nc; ++c) {
                    double *a = &S[head+(size_t)ldS*(j+c)];
                    for (t = 0; t < bcur; ++t) {
                        const double *vt = &S[head+(size_t)ldS*(k0-1+t)];
                        double s = a[t]; int ht = Ht[t];
                        for (i = t+1; i < ht; ++i) s += vt[i]*a[i];
                        w[t] = s; fl += 2.0*(double)(ht-t);
                    }
                    for (t = 0; t < bcur; ++t) {
                        double s = 0.0; int u;
                        for (u = 0; u <= t; ++u) s += T[u + (size_t)blk*t]*w[u];
                        w2[t] = s; fl += 2.0*(double)(t+1);
                    }
                    for (t = 0; t < bcur; ++t) {
                        const double *vt = &S[head+(size_t)ldS*(k0-1+t)];
                        double wt = w2[t]; int ht = Ht[t];
                        a[t] -= wt;
                        for (i = t+1; i < ht; ++i) a[i] -= vt[i]*wt;
                        fl += 2.0*(double)(ht-t);
                    }
                    if (Rp) {
                        double *dst = SIDQR_RCOL(j+c) + (k0-1);
                        for (t = 0; t < bcur; ++t) dst[t] = a[t];
                        if (Rld && Rsym)
                            for (t = 0; t < bcur; ++t)
                                Rp[(j+c) + (size_t)Rld*(k0-1+t)] = a[t];
                    }
                    if (Y) {
                        int e; for (e = 0; e < nrhs; ++e) {
                            double *ye = Y + (size_t)m*e;
                            double acc = 0.0;
                            for (t = 0; t < bcur; ++t) acc += a[t]*ye[k0-1+t];
                            ye[j+c] -= acc;
                        }
                        fl += 2.0*(double)bcur*(double)nrhs;
                    }
                    for (i = bcur; i < H; ++i) a[i-bcur] = a[i];
                    for (i = H-bcur; i < Hnext; ++i) a[i] = 0.0;
                }
            }
            j += nc;
        }
    }
    st->t_fact      = wtime() - t0;
    st->flops       = fl;
    st->Hmax        = Hs;
    st->strip_alloc = (size_t)ldS * (size_t)m * sizeof(double);
    st->strip_peak  = peak * sizeof(double);
#ifdef SIDQR_USE_MWBLAS
    if (backend) {
        /* Actual allocated factorisation workspace, excluding packed input
           and R (reported separately).  For lazy-strip mode this includes the
           reserved 16-panel headroom in ldS, not only logically live rows. */
        st->strip_peak = ((size_t)ldS * (size_t)m
                         + (size_t)Hs * (size_t)blk
                         + 2u * (size_t)blk * (size_t)m
                         + (size_t)blk * (size_t)blk
                         + (size_t)blk
                         + 2u * (size_t)blk * SIDQR_NCB) * sizeof(double)
                         + (size_t)blk * sizeof(int);
    }
#endif
    st->R_bytes     = Rp ? ((size_t)m * (m + 1)) / 2 * sizeof(double) : 0;
    st->input_bytes = (size_t)m * (size_t)(nth + bet) * sizeof(double);

    free(S); free(T); free(tau); free(w); free(w2); free(Ht);
#ifdef SIDQR_USE_MWBLAS
    free(VB); free(WB); free(W2B);
#endif
    return 0;
}

int sidqr_hh(const sidqr_dims *d, const double *Th, const double *Bd,
             int blk, double *Rp, int Rld, sidqr_stats *st)
{
    return hh_core(d, Th, Bd, blk, Rp, Rld, 0, NULL, 0, st, 0);
}

#ifdef SIDQR_USE_MWBLAS
int sidqr_hh_blas(const sidqr_dims *d, const double *Th, const double *Bd,
                  int blk, double *Rp, int Rld, sidqr_stats *st)
{
    return hh_core(d, Th, Bd, blk, Rp, Rld, 0, NULL, 0, st, 1);
}

int sidqr_hh_blas_lazy(const sidqr_dims *d, const double *Th, const double *Bd,
                       int blk, double *Rp, int Rld, sidqr_stats *st)
{
    return hh_core(d, Th, Bd, blk, Rp, Rld, 0, NULL, 0, st, 2);
}
#endif

/* =========================================================================
 *  Triangular solves.
 *
 *  Column-packed storage  Rp[j(j+1)/2+i] = R(i,j), i<=j, is optimal for
 *  BOTH solves and needs no transposed copy:
 *
 *   - forward,  R^T y = b :  y_i = (b_i - sum_{j<i} R(j,i) y_j)/R(i,i);
 *     the sum is a dot product over the *contiguous* column i, and the
 *     columns are visited in increasing order  -> forward streaming;
 *   - backward, R x = y   :  x_j = y_j/R(j,j), then y_{0:j-1} -= R(0:j-1,j)x_j;
 *     an axpy over the contiguous column j, columns visited in decreasing
 *     order  -> backward streaming.
 *
 *  Each solve therefore reads m(m+1)/2 doubles exactly once, sequentially,
 *  and no transposed copy of R is ever needed.
 * ======================================================================= */

/* dot product with four accumulators.  Without it the dot-shaped solves are
   latency bound on the serial reduction and run 3x slower than the
   axpy-shaped ones -- an artefact of the loop shape, not of the storage. */
static double dot4(const double *a, const double *b, int n)
{
    double s0=0,s1=0,s2=0,s3=0; int i, n4 = n & ~3;
    for (i = 0; i < n4; i += 4) {
        s0 += a[i]*b[i];     s1 += a[i+1]*b[i+1];
        s2 += a[i+2]*b[i+2]; s3 += a[i+3]*b[i+3];
    }
    for (; i < n; ++i) s0 += a[i]*b[i];
    return (s0+s1)+(s2+s3);
}

void sidqr_trsv_fwd(int m, const double *Rp, double *y, int nrhs, double *flops)
{
    int i, e;
    for (e = 0; e < nrhs; ++e) {
        double *ye = y + (size_t)m * e;
        for (i = 0; i < m; ++i) {
            const double *col = Rp + ((size_t)i * (i + 1)) / 2;
            ye[i] = (ye[i] - dot4(col, ye, i)) / col[i];
        }
    }
    if (flops) *flops += (double)nrhs * (double)m * (double)m;
}

void sidqr_trsv_bwd(int m, const double *Rp, double *y, int nrhs, double *flops)
{
    int i, j, e;
    for (e = 0; e < nrhs; ++e) {
        double *ye = y + (size_t)m * e;
        for (j = m - 1; j >= 0; --j) {
            const double *col = Rp + ((size_t)j * (j + 1)) / 2;
            double xj = ye[j] / col[j];
            ye[j] = xj;
            for (i = 0; i < j; ++i) ye[i] -= col[i] * xj;
        }
    }
    if (flops) *flops += (double)nrhs * (double)m * (double)m;
}

/* Row oriented solves on a full m x m array holding R in the upper triangle
   and R^T in the lower one (leading dimension Rld).  "Row i of R" is then
   the contiguous run A[Rld*i+i .. Rld*i+m-1], so both solves are dot
   products over contiguous data.  This is the variant that trades twice the
   storage and twice the write traffic for a marginally different read
   pattern; see the report for the measurement. */
void sidqr_trsv_fwd_sym(int m, const double *A, int Rld, double *y,
                        int nrhs, double *flops)
{
    int i, e;
    for (e = 0; e < nrhs; ++e) {
        double *ye = y + (size_t)m * e;
        for (i = 0; i < m; ++i) {
            const double *row = A + (size_t)Rld * i;
            ye[i] = (ye[i] - dot4(row, ye, i)) / A[i + (size_t)Rld * i];
        }
    }
    if (flops) *flops += (double)nrhs * (double)m * (double)m;
}

void sidqr_trsv_bwd_sym(int m, const double *A, int Rld, double *y,
                        int nrhs, double *flops)
{
    int i, e;
    for (e = 0; e < nrhs; ++e) {
        double *ye = y + (size_t)m * e;
        for (i = m - 1; i >= 0; --i) {
            const double *row = A + (size_t)Rld * i;
            ye[i] = (ye[i] - dot4(row + i + 1, ye + i + 1, m - i - 1)) / row[i];
        }
    }
    if (flops) *flops += (double)nrhs * (double)m * (double)m;
}

/* =========================================================================
 *  Factorise and solve  (J J^T) x = b  for nrhs right-hand sides.
 *
 *  mode 0 : column-packed R, factorise then two separate triangular solves
 *           (three sequential passes over R: one write, two reads)
 *  mode 1 : column-packed R, forward substitution fused into the sweep,
 *           then one backward solve  (one write, one read)      <- best
 *  mode 2 : full m x m array holding R and R^T, fused forward substitution,
 *           row-oriented backward solve  (two writes, one read, 2x memory)
 * ======================================================================= */
static int solve_core(const sidqr_dims *d, const double *Th, const double *Bd,
                      int blk, const double *b, int nrhs, double *x,
                      int mode, sidqr_stats *st, int backend)
{
    const int m = d->m;
    double *Rp = NULL, t0, fl = 0.0;
    size_t nR;
    int rc, Rld = 0, Rsym = 0, e, i;

    if (mode == 2) { Rld = m; Rsym = 1; nR = (size_t)m * (size_t)m; }
    else           { nR = ((size_t)m * (m + 1)) / 2; }

    Rp = (double *)malloc(nR * sizeof(double));
    if (!Rp) return -2;

    for (e = 0; e < nrhs; ++e)
        for (i = 0; i < m; ++i) x[i + (size_t)m * e] = b[i + (size_t)m * e];

    if (mode == 0) {
        rc = hh_core(d, Th, Bd, blk, Rp, 0, 0, NULL, 0, st, backend);
        if (rc) { free(Rp); return rc; }
        t0 = wtime();
        sidqr_trsv_fwd(m, Rp, x, nrhs, &fl);
        sidqr_trsv_bwd(m, Rp, x, nrhs, &fl);
    } else {
        rc = hh_core(d, Th, Bd, blk, Rp, Rld, Rsym, x, nrhs, st, backend);
        if (rc) { free(Rp); return rc; }
        t0 = wtime();
        if (mode == 2) sidqr_trsv_bwd_sym(m, Rp, Rld, x, nrhs, &fl);
        else           sidqr_trsv_bwd(m, Rp, x, nrhs, &fl);
    }
    st->t_solve  = wtime() - t0;
    st->flops   += fl;
    st->R_bytes  = nR * sizeof(double);
    free(Rp);
    return 0;
}

int sidqr_solve(const sidqr_dims *d, const double *Th, const double *Bd,
                int blk, const double *b, int nrhs, double *x,
                int mode, sidqr_stats *st)
{
    return solve_core(d, Th, Bd, blk, b, nrhs, x, mode, st, 0);
}

#ifdef SIDQR_USE_MWBLAS
int sidqr_solve_blas(const sidqr_dims *d, const double *Th, const double *Bd,
                     int blk, const double *b, int nrhs, double *x,
                     int mode, sidqr_stats *st)
{
    return solve_core(d, Th, Bd, blk, b, nrhs, x, mode, st, 1);
}

int sidqr_solve_blas_lazy(const sidqr_dims *d, const double *Th, const double *Bd,
                          int blk, const double *b, int nrhs, double *x,
                          int mode, sidqr_stats *st)
{
    return solve_core(d, Th, Bd, blk, b, nrhs, x, mode, st, 2);
}
#endif

/* =========================================================================
 *  Matrix-free  v = (J J^T) u , straight from the packed storage.
 *  Used by the harnesses to certify a solution without ever forming the
 *  m x m Gram matrix: 4*m*(ntheta+beta) flops and O(nu) workspace.
 * ======================================================================= */
int sidqr_gram_mv(const sidqr_dims *d, const double *Th, const double *Bd,
                  const double *u, double *v)
{
    const int m = d->m, nth = d->ntheta, bet = d->beta;
    double *w = (double *)calloc((size_t)d->nu, sizeof(double));
    int j, i, top;
    if (!w) return -2;
    for (j = 0; j < m; ++j) {                    /* w = X u   ( = J^T u ) */
        double uj = u[j];
        if (uj == 0.0) continue;
        for (i = 0; i < nth; ++i) w[i] += Th[i + (size_t)nth * j] * uj;
        top = sidqr_btop(d, j + 1) - 1;
        for (i = 0; i < bet; ++i) w[top + i] += Bd[i + (size_t)bet * j] * uj;
    }
    for (j = 0; j < m; ++j) {                    /* v = X^T w ( = J w )   */
        double s = 0.0;
        for (i = 0; i < nth; ++i) s += Th[i + (size_t)nth * j] * w[i];
        top = sidqr_btop(d, j + 1) - 1;
        for (i = 0; i < bet; ++i) s += Bd[i + (size_t)bet * j] * w[top + i];
        v[j] = s;
    }
    free(w);
    return 0;
}

/* =========================================================================
 *  Dense structure-blind Householder QR -- reference, small sizes only.
 * ======================================================================= */
int sidqr_dense(const sidqr_dims *d, const double *Th, const double *Bd,
                double *Rp, int Rld, sidqr_stats *st)
{
    const int m = d->m, nu = d->nu, nth = d->ntheta, bet = d->beta, p = d->p;
    double *A, *tau, fl = 0.0, t0;
    int j, k, i, l;

    A = (double *)calloc((size_t)nu * (size_t)m, sizeof(double));
    tau = (double *)calloc((size_t)m, sizeof(double));
    if (!A || !tau) { free(A); free(tau); return -2; }

    for (j = 0; j < m; ++j) {
        for (i = 0; i < nth; ++i) A[i + (size_t)nu * j] = Th[i + (size_t)nth * j];
        l = j / p;                                    /* 0-based block */
        for (i = 0; i < bet; ++i)
            A[(nth + d->gamma * l + i) + (size_t)nu * j] = Bd[i + (size_t)bet * j];
    }
    t0 = wtime();
    for (k = 0; k < m; ++k) {
        double *x = &A[k + (size_t)nu * k];
        int len = nu - k;
        tau[k] = housegen(x, len, &fl);
        for (j = k + 1; j < m; ++j) {
            double *a = &A[(size_t)nu * j];
            double s = a[k];
            for (i = k + 1; i < nu; ++i) s += x[i - k] * a[i];
            s *= tau[k];
            a[k] -= s;
            for (i = k + 1; i < nu; ++i) a[i] -= x[i - k] * s;
            fl += 4.0 * (double)len;
        }
    }
    st->t_fact = wtime() - t0;
    st->flops  = fl;
    st->Hmax   = nu;
    st->strip_alloc = (size_t)nu * (size_t)m * sizeof(double);
    st->strip_peak  = st->strip_alloc;
    st->R_bytes     = Rp ? ((size_t)m * (m + 1)) / 2 * sizeof(double) : 0;
    st->input_bytes = (size_t)nu * (size_t)m * sizeof(double);
    if (Rp)
        for (j = 0; j < m; ++j) {
            double *dst = SIDQR_RCOL(j);
            for (i = 0; i <= j; ++i) dst[i] = A[i + (size_t)nu * j];
        }
    free(A); free(tau);
    return 0;
}

/* =========================================================================
 *  Exact operation counts (structural, no arithmetic).
 * ======================================================================= */
void sidqr_count(const sidqr_dims *d, double *fl_dw, double *fl_ex,
                 int *hmax, double *peak_strip)
{
    const int m = d->m;
    double dw = 0.0, ex = 0.0, pk = 0.0;
    int k, hM = 0;

    for (k = 1; k <= m; ++k) {
        int Bk  = sidqr_bbot(d, k);
        int Bk1 = (k > 1) ? sidqr_bbot(d, k - 1) : d->ntheta;   /* filled part */
        int h   = Bk - k + 1;
        int l;
        double nnzsum = 0.0;
        if (h > hM) hM = h;
        if ((double)h * (double)(m - k + 1) > pk) pk = (double)h * (m - k + 1);

        /* algorithm actually implemented: frontal window treated as dense   */
        dw += 3.0 * h + 4.0 * (double)h * (double)(m - k);

        /* theoretical count skipping every structural zero in the window    */
        {
            int filled = Bk1 - k + 1; if (filled < 0) filled = 0;
            for (l = k + 1; l <= m; ++l) {
                int bt = sidqr_btop(d, l);
                int extra = Bk - (bt - 1 > Bk1 ? bt - 1 : Bk1);
                if (extra < 0) extra = 0;
                nnzsum += (double)(filled + extra);
            }
            ex += 3.0 * h + 4.0 * nnzsum;
        }
    }
    if (fl_dw) *fl_dw = dw;
    if (fl_ex) *fl_ex = ex;
    if (hmax)  *hmax  = hM;
    if (peak_strip) *peak_strip = pk;
}
