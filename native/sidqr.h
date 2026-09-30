/* ============================================================================
 *  sidqr.h -- Structured Q-less Householder QR for the constraint Jacobian
 *             arising in simulation-error-minimisation system identification.
 *
 *  The matrix factorised is  X = J^T  with  J = dh/dxi  the Jacobian of the
 *  N-n block constraints of the NIO / EIV identification problem.
 *
 *  Sizes (all "general", nothing hard-coded):
 *      N       number of samples
 *      n       dynamical order of the model
 *      p       number of outputs
 *      q       number of *free* input variables   (q = 0  -> output-error,
 *                                                  q = #inputs -> EIV)
 *      ntheta  number of model parameters
 *
 *  Derived:
 *      gamma = p + q            signal variables per time step
 *      beta  = (n+1)*gamma      height of the signal band of one column block
 *      m     = p*(N-n)          number of constraints  = columns of X
 *      nu    = ntheta + gamma*N number of variables    = rows    of X
 *
 *  Structured (minimal) storage of X, nnz = m*(ntheta+beta):
 *      Th : ntheta x m, column major, ld = ntheta   (dense parameter rows)
 *      Bd : beta   x m, column major, ld = beta     (signal band)
 *      column j (1-based) belongs to block  l = ceil(j/p),
 *      Bd(i,j) = X( ntheta + gamma*(l-1) + i , j ),   i = 1..beta
 *
 *  Output: R (m x m upper triangular), either column-packed or with an
 *  explicit leading dimension -- see sidqr_hh.
 *
 *  Supplementary material of "Vanishing-gradient-free ... system
 *  identification", revision 1.  Portable C99, no external dependency.
 * ==========================================================================*/
#ifndef SIDQR_H
#define SIDQR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int N, n, p, q, ntheta;
    int gamma;      /* p+q                    */
    int beta;       /* (n+1)*gamma            */
    int m;          /* p*(N-n)                */
    int nu;         /* ntheta + gamma*N       */
} sidqr_dims;

typedef struct {
    double flops;          /* useful floating point operations executed      */
    double t_fact;         /* wall time of the factorisation      [s]        */
    int    Hmax;           /* largest frontal (window) height                */
    size_t strip_alloc;    /* bytes allocated for the sliding strip          */
    size_t strip_peak;     /* bytes actually live at the peak                */
    size_t R_bytes;        /* bytes of the packed triangular factor          */
    size_t input_bytes;    /* bytes of the structured input (Th,Bd)          */
    double t_solve;        /* wall time of the triangular solve(s) [s]       */
} sidqr_stats;

/* fill d from the five primitive sizes; returns 0 on success */
int  sidqr_setdims(sidqr_dims *d, int N, int n, int p, int q, int ntheta);

/* global 1-based row index of the first band entry of column j (1-based) */
static inline int sidqr_btop(const sidqr_dims *d, int j)
{ return d->ntheta + d->gamma * (((j + d->p - 1) / d->p) - 1) + 1; }

/* global 1-based index of the last row that can be non-zero in column j */
static inline int sidqr_bbot(const sidqr_dims *d, int j)
{ return sidqr_btop(d, j) + d->beta - 1; }

/* pseudo-random, well-scaled structured test matrix */
void sidqr_gen(const sidqr_dims *d, double *Th, double *Bd, unsigned seed);

/* -------------------------------------------------------------------------
 * Blocked structured Householder QR.
 *   blk  : panel width (rounded up to a multiple of p; blk = p gives the
 *          unblocked, column-block by column-block variant)
 *   Rp   : triangular output, or NULL to skip forming R
 *   Rld  : 0    -> Rp is column-packed triangular, m(m+1)/2 doubles
 *          >0   -> Rp is a full array with leading dimension Rld (>= m)
 * ---------------------------------------------------------------------- */
int sidqr_hh(const sidqr_dims *d, const double *Th, const double *Bd,
             int blk, double *Rp, int Rld, sidqr_stats *st);

/* -------------------------------------------------------------------------
 * Triangular solves on the column-packed factor.  Both stream sequentially
 * over R and neither needs a transposed copy: the forward solve is a dot
 * product over the contiguous column i (columns visited forwards), the
 * backward solve an axpy over the contiguous column j (columns visited
 * backwards).
 * ---------------------------------------------------------------------- */
void sidqr_trsv_fwd(int m, const double *Rp, double *y, int nrhs, double *flops);
void sidqr_trsv_bwd(int m, const double *Rp, double *y, int nrhs, double *flops);

/* ... and the same on a full m x m array holding R in the upper and R^T in
 * the lower triangle (leading dimension Rld). */
void sidqr_trsv_fwd_sym(int m, const double *A, int Rld, double *y, int nrhs,
                        double *flops);
void sidqr_trsv_bwd_sym(int m, const double *A, int Rld, double *y, int nrhs,
                        double *flops);

/* -------------------------------------------------------------------------
 * Factorise and solve  (J J^T) x = b  for nrhs right-hand sides.
 *   mode 0 : packed R, factorise then forward + backward solve
 *   mode 1 : packed R, forward substitution fused into the sweep   <- best
 *   mode 2 : full m x m holding R and R^T, fused forward, row-wise backward
 * ---------------------------------------------------------------------- */
int sidqr_solve(const sidqr_dims *d, const double *Th, const double *Bd,
                int blk, const double *b, int nrhs, double *x,
                int mode, sidqr_stats *st);

/* BLAS-3 variants delegate the compact-WY trailing update to dgemm. */
#ifdef SIDQR_USE_MWBLAS
int sidqr_hh_blas(const sidqr_dims *d, const double *Th, const double *Bd,
                  int blk, double *Rp, int Rld, sidqr_stats *st);
int sidqr_solve_blas(const sidqr_dims *d, const double *Th, const double *Bd,
                     int blk, const double *b, int nrhs, double *x,
                     int mode, sidqr_stats *st);
int sidqr_hh_blas_lazy(const sidqr_dims *d, const double *Th, const double *Bd,
                       int blk, double *Rp, int Rld, sidqr_stats *st);
int sidqr_solve_blas_lazy(const sidqr_dims *d, const double *Th, const double *Bd,
                          int blk, const double *b, int nrhs, double *x,
                          int mode, sidqr_stats *st);
#endif

/* matrix-free  v = (J J^T) u  from the packed storage, for verification */
int sidqr_gram_mv(const sidqr_dims *d, const double *Th, const double *Bd,
                  const double *u, double *v);

/* Dense (structure-blind) Householder QR -- reference, small sizes only. */
int sidqr_dense(const sidqr_dims *d, const double *Th, const double *Bd,
                double *Rp, int Rld, sidqr_stats *st);

/* Exact sparsity-aware operation count (no arithmetic performed).
 *  fl_dense_window : count of the algorithm implemented in sidqr_hh
 *                    (frontal window treated as dense)
 *  fl_exact        : count when every structural zero inside the window is
 *                    also skipped (theoretical lower bound of the
 *                    column-by-column Householder scheme); O(m^2) loop
 *  hmax            : peak frontal height
 *  peak_strip      : peak number of live doubles in the working strip     */
void sidqr_count(const sidqr_dims *d, double *fl_dense_window,
                 double *fl_exact, int *hmax, double *peak_strip);

#ifdef __cplusplus
}
#endif
#endif /* SIDQR_H */
