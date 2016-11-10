// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cgo provides an interface to bindings for a C LAPACK library.
package cgo

import (
	"github.com/gonum/blas"
	"github.com/gonum/lapack"
	"github.com/gonum/lapack/cgo/lapacke"
)

// Copied from lapack/native. Keep in sync.
const (
	absIncNotOne    = "lapack: increment not one or negative one"
	badD            = "lapack: d has insufficient length"
	badDecompUpdate = "lapack: bad decomp update"
	badDiag         = "lapack: bad diag"
	badDims         = "lapack: bad input dimensions"
	badDirect       = "lapack: bad direct"
	badE            = "lapack: e has insufficient length"
	badEVComp       = "lapack: bad EVComp"
	badEVJob        = "lapack: bad EVJob"
	badEVSide       = "lapack: bad EVSide"
	badHowMany      = "lapack: bad HowMany"
	badIlo          = "lapack: ilo out of range"
	badIhi          = "lapack: ihi out of range"
	badIpiv         = "lapack: insufficient permutation length"
	badJob          = "lapack: bad Job"
	badLdA          = "lapack: index of a out of range"
	badNorm         = "lapack: bad norm"
	badPivot        = "lapack: bad pivot"
	badS            = "lapack: s has insufficient length"
	badShifts       = "lapack: bad shifts"
	badSide         = "lapack: bad side"
	badSlice        = "lapack: bad input slice length"
	badStore        = "lapack: bad store"
	badTau          = "lapack: tau has insufficient length"
	badTauQ         = "lapack: tauQ has insufficient length"
	badTauP         = "lapack: tauP has insufficient length"
	badTrans        = "lapack: bad trans"
	badUplo         = "lapack: illegal triangle"
	badWork         = "lapack: insufficient working memory"
	badWorkStride   = "lapack: insufficient working array stride"
	badZ            = "lapack: insufficient z length"
	kGTM            = "lapack: k > m"
	kGTN            = "lapack: k > n"
	kLT0            = "lapack: k < 0"
	mLTN            = "lapack: m < n"
	negDimension    = "lapack: negative matrix dimension"
	negZ            = "lapack: negative z value"
	nLT0            = "lapack: n < 0"
	nLTM            = "lapack: n < m"
	shortWork       = "lapack: working array shorter than declared"
)

func min(m, n int) int {
	if m < n {
		return m
	}
	return n
}

func max(m, n int) int {
	if m < n {
		return n
	}
	return m
}

// checkMatrix verifies the parameters of a matrix input.
// Copied from lapack/native. Keep in sync.
func checkMatrix(m, n int, a []float64, lda int) {
	if m < 0 {
		panic("lapack: has negative number of rows")
	}
	if n < 0 {
		panic("lapack: has negative number of columns")
	}
	if lda < n {
		panic("lapack: stride less than number of columns")
	}
	if len(a) < (m-1)*lda+n {
		panic("lapack: insufficient matrix slice length")
	}
}

// checkVector verifies the parameters of a vector input.
// Copied from lapack/native. Keep in sync.
func checkVector(n int, v []float64, inc int) {
	if n < 0 {
		panic("lapack: negative vector length")
	}
	if (inc > 0 && (n-1)*inc >= len(v)) || (inc < 0 && (1-n)*inc >= len(v)) {
		panic("lapack: insufficient vector slice length")
	}
}

// Implementation is the cgo-based C implementation of LAPACK routines.
type Implementation struct{}

var _ lapack.Float64 = Implementation{}

// Dlacn2 estimates the 1-norm of an n×n matrix A using sequential updates with
// matrix-vector products provided externally.
//
// Dlacn2 is called sequentially and it returns the value of est and kase to be
// used on the next call.
// On the initial call, kase must be 0.
// In between calls, x must be overwritten by
//  A * X    if kase was returned as 1,
//  A^T * X  if kase was returned as 2,
// and all other parameters must not be changed.
// On the final return, kase is returned as 0, v contains A*W where W is a
// vector, and est = norm(V)/norm(W) is a lower bound for 1-norm of A.
//
// v, x, and isgn must all have length n and n must be at least 1, otherwise
// Dlacn2 will panic. isave is used for temporary storage.
//
// Dlacn2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlacn2(n int, v, x []float64, isgn []int, est float64, kase int, isave *[3]int) (float64, int) {
	if n < 1 {
		panic("lapack: non-positive n")
	}
	checkVector(n, x, 1)
	checkVector(n, v, 1)
	if len(isgn) < n {
		panic("lapack: insufficient isgn length")
	}
	if isave[0] < 0 || isave[0] > 5 {
		panic("lapack: bad isave value")
	}
	if isave[0] == 0 && kase != 0 {
		panic("lapack: bad isave value")
	}

	isgn32 := make([]int32, n)
	for i, v := range isgn {
		isgn32[i] = int32(v)
	}
	pest := []float64{est}
	// Save one allocation by putting isave and kase into the same slice.
	isavekase := []int32{int32(isave[0]), int32(isave[1]), int32(isave[2]), int32(kase)}
	lapacke.Dlacn2(n, v, x, isgn32, pest, isavekase[3:], isavekase[:3])
	for i, v := range isgn32 {
		isgn[i] = int(v)
	}
	isave[0] = int(isavekase[0])
	isave[1] = int(isavekase[1])
	isave[2] = int(isavekase[2])

	return pest[0], int(isavekase[3])
}

// Dlacpy copies the elements of A specified by uplo into B. Uplo can specify
// a triangular portion with blas.Upper or blas.Lower, or can specify all of the
// elemest with blas.All.
func (impl Implementation) Dlacpy(uplo blas.Uplo, m, n int, a []float64, lda int, b []float64, ldb int) {
	checkMatrix(m, n, a, lda)
	checkMatrix(m, n, b, ldb)
	lapacke.Dlacpy(uplo, m, n, a, lda, b, ldb)
}

// Dlarfb applies a block reflector to a matrix.
//
// In the call to Dlarfb, the mxn c is multiplied by the implicitly defined matrix h as follows:
//  c = h * c if side == Left and trans == NoTrans
//  c = c * h if side == Right and trans == NoTrans
//  c = h^T * c if side == Left and trans == Trans
//  c = c * h^T if side == Right and trans == Trans
// h is a product of elementary reflectors. direct sets the direction of multiplication
//  h = h_1 * h_2 * ... * h_k if direct == Forward
//  h = h_k * h_k-1 * ... * h_1 if direct == Backward
// The combination of direct and store defines the orientation of the elementary
// reflectors. In all cases the ones on the diagonal are implicitly represented.
//
// If direct == lapack.Forward and store == lapack.ColumnWise
//  V = [ 1        ]
//      [v1   1    ]
//      [v1  v2   1]
//      [v1  v2  v3]
//      [v1  v2  v3]
// If direct == lapack.Forward and store == lapack.RowWise
//  V = [ 1  v1  v1  v1  v1]
//      [     1  v2  v2  v2]
//      [         1  v3  v3]
// If direct == lapack.Backward and store == lapack.ColumnWise
//  V = [v1  v2  v3]
//      [v1  v2  v3]
//      [ 1  v2  v3]
//      [     1  v3]
//      [         1]
// If direct == lapack.Backward and store == lapack.RowWise
//  V = [v1  v1   1        ]
//      [v2  v2  v2   1    ]
//      [v3  v3  v3  v3   1]
// An elementary reflector can be explicitly constructed by extracting the
// corresponding elements of v, placing a 1 where the diagonal would be, and
// placing zeros in the remaining elements.
//
// t is a k×k matrix containing the block reflector, and this function will panic
// if t is not of sufficient size. See Dlarft for more information.
//
// work is a temporary storage matrix with stride ldwork.
// work must be of size at least n×k side == Left and m×k if side == Right, and
// this function will panic if this size is not met.
//
// Dlarfb is an internal routine. It is exported for testing purposes.
func (Implementation) Dlarfb(side blas.Side, trans blas.Transpose, direct lapack.Direct,
	store lapack.StoreV, m, n, k int, v []float64, ldv int, t []float64, ldt int,
	c []float64, ldc int, work []float64, ldwork int) {

	checkMatrix(m, n, c, ldc)
	if m == 0 || n == 0 {
		return
	}
	if k < 0 {
		panic("lapack: negative number of transforms")
	}
	if side != blas.Left && side != blas.Right {
		panic(badSide)
	}
	if trans != blas.Trans && trans != blas.NoTrans {
		panic(badTrans)
	}
	if direct != lapack.Forward && direct != lapack.Backward {
		panic(badDirect)
	}
	if store != lapack.ColumnWise && store != lapack.RowWise {
		panic(badStore)
	}

	rowsWork := n
	if side == blas.Right {
		rowsWork = m
	}
	// TODO(vladimir-ch): Replace the following two lines with
	//  checkMatrix(rowsWork, k, work, ldwork)
	// if and when the issue
	//  https://github.com/Reference-LAPACK/lapack/issues/37
	// has been resolved.
	ldwork = rowsWork
	work = make([]float64, ldwork*k)

	lapacke.Dlarfb(side, trans, byte(direct), byte(store), m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork)
}

// Dlarfg generates an elementary reflector for a Householder matrix. It creates
// a real elementary reflector of order n such that
//  H * (alpha) = (beta)
//      (    x)   (   0)
//  H^T * H = I
// H is represented in the form
//  H = 1 - tau * (1; v) * (1 v^T)
// where tau is a real scalar.
//
// On entry, x contains the vector x, on exit it contains v.
//
// Dlarfg is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlarfg(n int, alpha float64, x []float64, incX int) (beta, tau float64) {
	if n < 0 {
		panic(nLT0)
	}
	if n <= 1 {
		return alpha, 0
	}
	checkVector(n-1, x, incX)
	_alpha := []float64{alpha}
	_tau := []float64{0}
	lapacke.Dlarfg(n, _alpha, x, incX, _tau)
	return _alpha[0], _tau[0]
}

// Dlarft forms the triangular factor T of a block reflector H, storing the answer
// in t.
//  H = I - V * T * V^T  if store == lapack.ColumnWise
//  H = I - V^T * T * V  if store == lapack.RowWise
// H is defined by a product of the elementary reflectors where
//  H = H_0 * H_1 * ... * H_{k-1}  if direct == lapack.Forward
//  H = H_{k-1} * ... * H_1 * H_0  if direct == lapack.Backward
//
// t is a k×k triangular matrix. t is upper triangular if direct = lapack.Forward
// and lower triangular otherwise. This function will panic if t is not of
// sufficient size.
//
// store describes the storage of the elementary reflectors in v. Please see
// Dlarfb for a description of layout.
//
// tau contains the scalar factors of the elementary reflectors H_i.
//
// Dlarft is an internal routine. It is exported for testing purposes.
func (Implementation) Dlarft(direct lapack.Direct, store lapack.StoreV, n, k int,
	v []float64, ldv int, tau []float64, t []float64, ldt int) {
	if n == 0 {
		return
	}
	if n < 0 || k < 0 {
		panic(negDimension)
	}
	if direct != lapack.Forward && direct != lapack.Backward {
		panic(badDirect)
	}
	if store != lapack.RowWise && store != lapack.ColumnWise {
		panic(badStore)
	}
	if len(tau) < k {
		panic(badTau)
	}
	checkMatrix(k, k, t, ldt)

	lapacke.Dlarft(byte(direct), byte(store), n, k, v, ldv, tau, t, ldt)
}

// Dlange computes the matrix norm of the general m×n matrix a. The input norm
// specifies the norm computed.
//  lapack.MaxAbs: the maximum absolute value of an element.
//  lapack.MaxColumnSum: the maximum column sum of the absolute values of the entries.
//  lapack.MaxRowSum: the maximum row sum of the absolute values of the entries.
//  lapack.Frobenius: the square root of the sum of the squares of the entries.
// If norm == lapack.MaxColumnSum, work must be of length n, and this function will panic otherwise.
// There are no restrictions on work for the other matrix norms.
func (impl Implementation) Dlange(norm lapack.MatrixNorm, m, n int, a []float64, lda int, work []float64) float64 {
	checkMatrix(m, n, a, lda)
	switch norm {
	case lapack.MaxRowSum, lapack.MaxColumnSum, lapack.NormFrob, lapack.MaxAbs:
	default:
		panic(badNorm)
	}
	if norm == lapack.MaxColumnSum && len(work) < n {
		panic(badWork)
	}
	return lapacke.Dlange(byte(norm), m, n, a, lda, work)
}

// Dlansy computes the specified norm of an n×n symmetric matrix. If
// norm == lapack.MaxColumnSum or norm == lapackMaxRowSum work must have length
// at least n, otherwise work is unused.
func (impl Implementation) Dlansy(norm lapack.MatrixNorm, uplo blas.Uplo, n int, a []float64, lda int, work []float64) float64 {
	checkMatrix(n, n, a, lda)
	switch norm {
	case lapack.MaxRowSum, lapack.MaxColumnSum, lapack.NormFrob, lapack.MaxAbs:
	default:
		panic(badNorm)
	}
	if (norm == lapack.MaxColumnSum || norm == lapack.MaxRowSum) && len(work) < n {
		panic(badWork)
	}
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	return lapacke.Dlansy(byte(norm), uplo, n, a, lda, work)
}

// Dlantr computes the specified norm of an m×n trapezoidal matrix A. If
// norm == lapack.MaxColumnSum work must have length at least n, otherwise work
// is unused.
func (impl Implementation) Dlantr(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, m, n int, a []float64, lda int, work []float64) float64 {
	checkMatrix(m, n, a, lda)
	switch norm {
	case lapack.MaxRowSum, lapack.MaxColumnSum, lapack.NormFrob, lapack.MaxAbs:
	default:
		panic(badNorm)
	}
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	if norm == lapack.MaxColumnSum && len(work) < n {
		panic(badWork)
	}
	return lapacke.Dlantr(byte(norm), uplo, diag, m, n, a, lda, work)
}

// Dlarfx applies an elementary reflector H to a real m×n matrix C, from either
// the left or the right, with loop unrolling when the reflector has order less
// than 11.
//
// H is represented in the form
//  H = I - tau * v * v^T,
// where tau is a real scalar and v is a real vector. If tau = 0, then H is
// taken to be the identity matrix.
//
// v must have length equal to m if side == blas.Left, and equal to n if side ==
// blas.Right, otherwise Dlarfx will panic.
//
// c and ldc represent the m×n matrix C. On return, C is overwritten by the
// matrix H * C if side == blas.Left, or C * H if side == blas.Right.
//
// work must have length at least n if side == blas.Left, and at least m if side
// == blas.Right, otherwise Dlarfx will panic. work is not referenced if H has
// order < 11.
func (impl Implementation) Dlarfx(side blas.Side, m, n int, v []float64, tau float64, c []float64, ldc int, work []float64) {
	checkMatrix(m, n, c, ldc)
	switch side {
	case blas.Left:
		checkVector(m, v, 1)
		if len(work) < n && m > 10 {
			panic(badWork)
		}
	case blas.Right:
		checkVector(n, v, 1)
		if len(work) < m && n > 10 {
			panic(badWork)
		}
	default:
		panic(badSide)
	}

	lapacke.Dlarfx(side, m, n, v, tau, c, ldc, work)
}

// Dpotrf computes the Cholesky decomposition of the symmetric positive definite
// matrix a. If ul == blas.Upper, then a is stored as an upper-triangular matrix,
// and a = U U^T is stored in place into a. If ul == blas.Lower, then a = L L^T
// is computed and stored in-place into a. If a is not positive definite, false
// is returned. This is the blocked version of the algorithm.
func (impl Implementation) Dpotrf(ul blas.Uplo, n int, a []float64, lda int) (ok bool) {
	// ul is checked in lapacke.Dpotrf.
	if n < 0 {
		panic(nLT0)
	}
	if lda < n {
		panic(badLdA)
	}
	if n == 0 {
		return true
	}
	return lapacke.Dpotrf(ul, n, a, lda)
}

// Dgebal balances an n×n matrix A. Balancing consists of two stages, permuting
// and scaling. Both steps are optional and depend on the value of job.
//
// Permuting consists of applying a permutation matrix P such that the matrix
// that results from P^T*A*P takes the upper block triangular form
//            [ T1  X  Y  ]
//  P^T A P = [  0  B  Z  ],
//            [  0  0  T2 ]
// where T1 and T2 are upper triangular matrices and B contains at least one
// nonzero off-diagonal element in each row and column. The indices ilo and ihi
// mark the starting and ending columns of the submatrix B. The eigenvalues of A
// isolated in the first 0 to ilo-1 and last ihi+1 to n-1 elements on the
// diagonal can be read off without any roundoff error.
//
// Scaling consists of applying a diagonal similarity transformation D such that
// D^{-1}*B*D has the 1-norm of each row and its corresponding column nearly
// equal. The output matrix is
//  [ T1     X*D          Y    ]
//  [  0  inv(D)*B*D  inv(D)*Z ].
//  [  0      0           T2   ]
// Scaling may reduce the 1-norm of the matrix, and improve the accuracy of
// the computed eigenvalues and/or eigenvectors.
//
// job specifies the operations that will be performed on A.
// If job is lapack.None, Dgebal sets scale[i] = 1 for all i and returns ilo=0, ihi=n-1.
// If job is lapack.Permute, only permuting will be done.
// If job is lapack.Scale, only scaling will be done.
// If job is lapack.PermuteScale, both permuting and scaling will be done.
//
// On return, if job is lapack.Permute or lapack.PermuteScale, it will hold that
//  A[i,j] == 0,   for i > j and j ∈ {0, ..., ilo-1, ihi+1, ..., n-1}.
// If job is lapack.None or lapack.Scale, or if n == 0, it will hold that
//  ilo == 0 and ihi == n-1.
//
// On return, scale will contain information about the permutations and scaling
// factors applied to A. If π(j) denotes the index of the column interchanged
// with column j, and D[j,j] denotes the scaling factor applied to column j,
// then
//  scale[j] == π(j),     for j ∈ {0, ..., ilo-1, ihi+1, ..., n-1},
//           == D[j,j],   for j ∈ {ilo, ..., ihi}.
// scale must have length equal to n, otherwise Dgebal will panic.
//
// Dgebal is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebal(job lapack.Job, n int, a []float64, lda int, scale []float64) (ilo, ihi int) {
	switch job {
	default:
		panic(badJob)
	case lapack.None, lapack.Permute, lapack.Scale, lapack.PermuteScale:
	}
	checkMatrix(n, n, a, lda)
	if len(scale) != n {
		panic("lapack: bad length of scale")
	}

	ilo32 := make([]int32, 1)
	ihi32 := make([]int32, 1)
	lapacke.Dgebal(job, n, a, lda, ilo32, ihi32, scale)
	ilo = int(ilo32[0]) - 1
	ihi = int(ihi32[0]) - 1
	for j := 0; j < ilo; j++ {
		scale[j]--
	}
	for j := ihi + 1; j < n; j++ {
		scale[j]--
	}
	return ilo, ihi
}

// Dgebak transforms an n×m matrix V as
//  V = P D V,        if side == blas.Right,
//  V = P D^{-1} V,   if side == blas.Left,
// where P and D are n×n permutation and scaling matrices, respectively,
// implicitly represented by job, scale, ilo and ihi as returned by Dgebal.
//
// Typically, columns of the matrix V contain the right or left (determined by
// side) eigenvectors of the balanced matrix output by Dgebal, and Dgebak forms
// the eigenvectors of the original matrix.
//
// Dgebak is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebak(job lapack.Job, side lapack.EVSide, n, ilo, ihi int, scale []float64, m int, v []float64, ldv int) {
	switch job {
	default:
		panic(badJob)
	case lapack.None, lapack.Permute, lapack.Scale, lapack.PermuteScale:
	}
	var bside blas.Side
	switch side {
	default:
		panic(badSide)
	case lapack.LeftEV:
		bside = blas.Left
	case lapack.RightEV:
		bside = blas.Right
	}
	checkMatrix(n, m, v, ldv)
	switch {
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	}

	// Convert permutation indices to 1-based.
	for j := 0; j < ilo; j++ {
		scale[j]++
	}
	for j := ihi + 1; j < n; j++ {
		scale[j]++
	}
	lapacke.Dgebak(job, bside, n, ilo+1, ihi+1, scale, m, v, ldv)
	// Convert permutation indices back to 0-based.
	for j := 0; j < ilo; j++ {
		scale[j]--
	}
	for j := ihi + 1; j < n; j++ {
		scale[j]--
	}
}

// Dbdsqr performs a singular value decomposition of a real n×n bidiagonal matrix.
//
// The SVD of the bidiagonal matrix B is
//  B = Q * S * P^T
// where S is a diagonal matrix of singular values, Q is an orthogonal matrix of
// left singular vectors, and P is an orthogonal matrix of right singular vectors.
//
// Q and P are only computed if requested. If left singular vectors are requested,
// this routine returns U * Q instead of Q, and if right singular vectors are
// requested P^T * VT is returned instead of P^T.
//
// Frequently Dbdsqr is used in conjunction with Dgebrd which reduces a general
// matrix A into bidiagonal form. In this case, the SVD of A is
//  A = (U * Q) * S * (P^T * VT)
//
// This routine may also compute Q^T * C.
//
// d and e contain the elements of the bidiagonal matrix b. d must have length at
// least n, and e must have length at least n-1. Dbdsqr will panic if there is
// insufficient length. On exit, D contains the singular values of B in decreasing
// order.
//
// VT is a matrix of size n×ncvt whose elements are stored in vt. The elements
// of vt are modified to contain P^T * VT on exit. VT is not used if ncvt == 0.
//
// U is a matrix of size nru×n whose elements are stored in u. The elements
// of u are modified to contain U * Q on exit. U is not used if nru == 0.
//
// C is a matrix of size n×ncc whose elements are stored in c. The elements
// of c are modified to contain Q^T * C on exit. C is not used if ncc == 0.
//
// work contains temporary storage and must have length at least 4*n. Dbdsqr
// will panic if there is insufficient working memory.
//
// Dbdsqr returns whether the decomposition was successful.
func (impl Implementation) Dbdsqr(uplo blas.Uplo, n, ncvt, nru, ncc int, d, e, vt []float64, ldvt int, u []float64, ldu int, c []float64, ldc int, work []float64) (ok bool) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if ncvt != 0 {
		checkMatrix(n, ncvt, vt, ldvt)
	}
	if nru != 0 {
		checkMatrix(nru, n, u, ldu)
	}
	if ncc != 0 {
		checkMatrix(n, ncc, c, ldc)
	}
	if len(d) < n {
		panic(badD)
	}
	if len(e) < n-1 {
		panic(badE)
	}
	if len(work) < 4*n {
		panic(badWork)
	}
	// An address must be passed to cgo. If lengths are zero, allocate a slice.
	if len(vt) == 0 {
		vt = make([]float64, 1)
	}
	if len(u) == 0 {
		vt = make([]float64, 1)
	}
	if len(c) == 0 {
		c = make([]float64, 1)
	}
	return lapacke.Dbdsqr(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work)
}

// Dgebrd reduces a general m×n matrix A to upper or lower bidiagonal form B by
// an orthogonal transformation:
//  Q^T * A * P = B.
// The diagonal elements of B are stored in d and the off-diagonal elements are
// stored in e. These are additionally stored along the diagonal of A and the
// off-diagonal of A. If m >= n B is an upper-bidiagonal matrix, and if m < n B
// is a lower-bidiagonal matrix.
//
// The remaining elements of A store the data needed to construct Q and P.
// The matrices Q and P are products of elementary reflectors
//  if m >= n, Q = H_0 * H_1 * ... * H_{n-1},
//             P = G_0 * G_1 * ... * G_{n-2},
//  if m < n,  Q = H_0 * H_1 * ... * H_{m-2},
//             P = G_0 * G_1 * ... * G_{m-1},
// where
//  H_i = I - tauQ[i] * v_i * v_i^T,
//  G_i = I - tauP[i] * u_i * u_i^T.
//
// As an example, on exit the entries of A when m = 6, and n = 5
//  [ d   e  u1  u1  u1]
//  [v1   d   e  u2  u2]
//  [v1  v2   d   e  u3]
//  [v1  v2  v3   d   e]
//  [v1  v2  v3  v4   d]
//  [v1  v2  v3  v4  v5]
// and when m = 5, n = 6
//  [ d  u1  u1  u1  u1  u1]
//  [ e   d  u2  u2  u2  u2]
//  [v1   e   d  u3  u3  u3]
//  [v1  v2   e   d  u4  u4]
//  [v1  v2  v3   e   d  u5]
//
// d, tauQ, and tauP must all have length at least min(m,n), and e must have
// length min(m,n) - 1.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(m,n) and this function will panic otherwise.
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgebrd but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dgebrd will panic.
func (impl Implementation) Dgebrd(m, n int, a []float64, lda int, d, e, tauQ, tauP, work []float64, lwork int) {
	checkMatrix(m, n, a, lda)
	minmn := min(m, n)
	if len(d) < minmn {
		panic(badD)
	}
	if len(e) < minmn-1 {
		panic(badE)
	}
	if len(tauQ) < minmn {
		panic(badTauQ)
	}
	if len(tauP) < minmn {
		panic(badTauP)
	}
	ws := max(m, n)
	if lwork == -1 {
		work[0] = float64(ws)
		return
	}
	if lwork < ws {
		panic(badWork)
	}
	if len(work) < lwork {
		panic(badWork)
	}

	lapacke.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, lwork)
}

// Dgecon estimates the reciprocal of the condition number of the n×n matrix A
// given the LU decomposition of the matrix. The condition number computed may
// be based on the 1-norm or the ∞-norm.
//
// The slice a contains the result of the LU decomposition of A as computed by Dgetrf.
//
// anorm is the corresponding 1-norm or ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 4*n and Dgecon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dgecon will panic otherwise.
// Elements of iwork must fit within the int32 type or Dgecon will panic.
func (impl Implementation) Dgecon(norm lapack.MatrixNorm, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64 {
	checkMatrix(n, n, a, lda)
	if norm != lapack.MaxColumnSum && norm != lapack.MaxRowSum {
		panic("bad norm")
	}
	if len(work) < 4*n {
		panic(badWork)
	}
	if len(iwork) < n {
		panic(badWork)
	}
	rcond := make([]float64, 1)
	_iwork := make([]int32, len(iwork))
	for i, v := range iwork {
		if v != int(int32(v)) {
			panic("lapack: iwork element out of range")
		}
		_iwork[i] = int32(v)
	}
	lapacke.Dgecon(byte(norm), n, a, lda, anorm, rcond, work, _iwork)
	for i, v := range _iwork {
		iwork[i] = int(v)
	}
	return rcond[0]
}

// Dgelq2 computes the LQ factorization of the m×n matrix A.
//
// In an LQ factorization, L is a lower triangular m×n matrix, and Q is an n×n
// orthonormal matrix.
//
// a is modified to contain the information to construct L and Q.
// The lower triangle of a contains the matrix L. The upper triangular elements
// (not including the diagonal) contain the elementary reflectors. Tau is modified
// to contain the reflector scales. tau must have length of at least k = min(m,n)
// and this function will panic otherwise.
//
// See Dgeqr2 for a description of the elementary reflectors and orthonormal
// matrix Q. Q is constructed as a product of these elementary reflectors,
//  Q = H_{k-1} * ... * H_1 * H_0,
// where k = min(m,n).
//
// Work is temporary storage of length at least m and this function will panic otherwise.
func (impl Implementation) Dgelq2(m, n int, a []float64, lda int, tau, work []float64) {
	checkMatrix(m, n, a, lda)
	if len(tau) < min(m, n) {
		panic(badTau)
	}
	if len(work) < m {
		panic(badWork)
	}
	lapacke.Dgelq2(m, n, a, lda, tau, work)
}

// Dgelqf computes the LQ factorization of the m×n matrix A using a blocked
// algorithm. See the documentation for Dgelq2 for a description of the
// parameters at entry and exit.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgelqf but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dgelqf will panic.
//
// tau must have length at least min(m,n), and this function will panic otherwise.
func (impl Implementation) Dgelqf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	if lwork == -1 {
		work[0] = float64(m)
		return
	}
	checkMatrix(m, n, a, lda)
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork < m {
		panic(badWork)
	}
	if len(tau) < min(m, n) {
		panic(badTau)
	}
	lapacke.Dgelqf(m, n, a, lda, tau, work, lwork)
}

// Dgeqr2 computes a QR factorization of the m×n matrix A.
//
// In a QR factorization, Q is an m×m orthonormal matrix, and R is an
// upper triangular m×n matrix.
//
// A is modified to contain the information to construct Q and R.
// The upper triangle of a contains the matrix R. The lower triangular elements
// (not including the diagonal) contain the elementary reflectors. Tau is modified
// to contain the reflector scales. tau must have length at least min(m,n), and
// this function will panic otherwise.
//
// The ith elementary reflector can be explicitly constructed by first extracting
// the
//  v[j] = 0           j < i
//  v[j] = 1           j == i
//  v[j] = a[j*lda+i]  j > i
// and computing H_i = I - tau[i] * v * v^T.
//
// The orthonormal matrix Q can be constucted from a product of these elementary
// reflectors, Q = H_0 * H_1 * ... * H_{k-1}, where k = min(m,n).
//
// Work is temporary storage of length at least n and this function will panic otherwise.
func (impl Implementation) Dgeqr2(m, n int, a []float64, lda int, tau, work []float64) {
	checkMatrix(m, n, a, lda)
	if len(work) < n {
		panic(badWork)
	}
	k := min(m, n)
	if len(tau) < k {
		panic(badTau)
	}
	lapacke.Dgeqr2(m, n, a, lda, tau, work)
}

// Dgeqrf computes the QR factorization of the m×n matrix A using a blocked
// algorithm. See the documentation for Dgeqr2 for a description of the
// parameters at entry and exit.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgeqrf but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dgeqrf will panic.
//
// tau must have length at least min(m,n), and this function will panic otherwise.
func (impl Implementation) Dgeqrf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	if lwork == -1 {
		work[0] = float64(n)
		return
	}
	checkMatrix(m, n, a, lda)
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork < n {
		panic(badWork)
	}
	k := min(m, n)
	if len(tau) < k {
		panic(badTau)
	}
	lapacke.Dgeqrf(m, n, a, lda, tau, work, lwork)
}

// Dgehrd reduces a block of a real n×n general matrix A to upper Hessenberg
// form H by an orthogonal similarity transformation Q^T * A * Q = H.
//
// The matrix Q is represented as a product of (ihi-ilo) elementary
// reflectors
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
// Each H_i has the form
//  H_i = I - tau[i] * v * v^T
// where v is a real vector with v[0:i+1] = 0, v[i+1] = 1 and v[ihi+1:n] = 0.
// v[i+2:ihi+1] is stored on exit in A[i+2:ihi+1,i].
//
// On entry, a contains the n×n general matrix to be reduced. On return, the
// upper triangle and the first subdiagonal of A will be overwritten with the
// upper Hessenberg matrix H, and the elements below the first subdiagonal, with
// the slice tau, represent the orthogonal matrix Q as a product of elementary
// reflectors.
//
// The contents of a are illustrated by the following example, with n = 7, ilo =
// 1 and ihi = 5.
// On entry,
//  [ a   a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [                         a ]
// on return,
//  [ a   a   h   h   h   h   a ]
//  [     a   h   h   h   h   a ]
//  [     h   h   h   h   h   h ]
//  [     v1  h   h   h   h   h ]
//  [     v1  v2  h   h   h   h ]
//  [     v1  v2  v3  h   h   h ]
//  [                         a ]
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H_i.
//
// ilo and ihi determine the block of A that will be reduced to upper Hessenberg
// form. It must hold that 0 <= ilo <= ihi < n if n > 0, and ilo == 0 and ihi ==
// -1 if n == 0, otherwise Dgehrd will panic.
//
// On return, tau will contain the scalar factors of the elementary reflectors.
// Elements tau[:ilo] and tau[ihi:] will be set to zero. tau must have length
// equal to n-1 if n > 0, otherwise Dgehrd will panic.
//
// work must have length at least lwork and lwork must be at least max(1,n),
// otherwise Dgehrd will panic. On return, work[0] contains the optimal value of
// lwork.
//
// If lwork == -1, instead of performing Dgehrd, only the optimal value of lwork
// will be stored in work[0].
//
// Dgehrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgehrd(n, ilo, ihi int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lwork < max(1, n) && lwork != -1:
		panic(badWork)
	case len(work) < lwork:
		panic(shortWork)
	}
	if lwork != -1 {
		checkMatrix(n, n, a, lda)
		if len(tau) != n-1 && n > 0 {
			panic(badTau)
		}
	}
	lapacke.Dgehrd(n, ilo+1, ihi+1, a, lda, tau, work, lwork)
}

// Dgels finds a minimum-norm solution based on the matrices A and B using the
// QR or LQ factorization. Dgels returns false if the matrix
// A is singular, and true if this solution was successfully found.
//
// The minimization problem solved depends on the input parameters.
//
//  1. If m >= n and trans == blas.NoTrans, Dgels finds X such that || A*X - B||_2
//     is minimized.
//  2. If m < n and trans == blas.NoTrans, Dgels finds the minimum norm solution of
//     A * X = B.
//  3. If m >= n and trans == blas.Trans, Dgels finds the minimum norm solution of
//     A^T * X = B.
//  4. If m < n and trans == blas.Trans, Dgels finds X such that || A*X - B||_2
//     is minimized.
// Note that the least-squares solutions (cases 1 and 3) perform the minimization
// per column of B. This is not the same as finding the minimum-norm matrix.
//
// The matrix A is a general matrix of size m×n and is modified during this call.
// The input matrix B is of size max(m,n)×nrhs, and serves two purposes. On entry,
// the elements of b specify the input matrix B. B has size m×nrhs if
// trans == blas.NoTrans, and n×nrhs if trans == blas.Trans. On exit, the
// leading submatrix of b contains the solution vectors X. If trans == blas.NoTrans,
// this submatrix is of size n×nrhs, and of size m×nrhs otherwise.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgels but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dgels will panic.
func (impl Implementation) Dgels(trans blas.Transpose, m, n, nrhs int, a []float64, lda int, b []float64, ldb int, work []float64, lwork int) bool {
	mn := min(m, n)
	if lwork == -1 {
		work[0] = float64(mn + max(mn, nrhs))
		return true
	}
	checkMatrix(m, n, a, lda)
	checkMatrix(max(m, n), nrhs, b, ldb)
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork < mn+max(mn, nrhs) {
		panic(badWork)
	}
	return lapacke.Dgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork)
}

const noSVDO = "dgesvd: not coded for overwrite"

// Dgesvd computes the singular value decomposition of the input matrix A.
//
// The singular value decomposition is
//  A = U * Sigma * V^T
// where Sigma is an m×n diagonal matrix containing the singular values of A,
// U is an m×m orthogonal matrix and V is an n×n orthogonal matrix. The first
// min(m,n) columns of U and V are the left and right singular vectors of A
// respectively.
//
// jobU and jobVT are options for computing the singular vectors. The behavior
// is as follows
//  jobU == lapack.SVDAll       All m columns of U are returned in u
//  jobU == lapack.SVDInPlace   The first min(m,n) columns are returned in u
//  jobU == lapack.SVDOverwrite The first min(m,n) columns of U are written into a
//  jobU == lapack.SVDNone      The columns of U are not computed.
// The behavior is the same for jobVT and the rows of V^T. At most one of jobU
// and jobVT can equal lapack.SVDOverwrite, and Dgesvd will panic otherwise.
//
// On entry, a contains the data for the m×n matrix A. During the call to Dgesvd
// the data is overwritten. On exit, A contains the appropriate singular vectors
// if either job is lapack.SVDOverwrite.
//
// s is a slice of length at least min(m,n) and on exit contains the singular
// values in decreasing order.
//
// u contains the left singular vectors on exit, stored column-wise. If
// jobU == lapack.SVDAll, u is of size m×m. If jobU == lapack.SVDInPlace u is
// of size m×min(m,n). If jobU == lapack.SVDOverwrite or lapack.SVDNone, u is
// not used.
//
// vt contains the left singular vectors on exit, stored rowwise. If
// jobV == lapack.SVDAll, vt is of size n×m. If jobVT == lapack.SVDInPlace vt is
// of size min(m,n)×n. If jobVT == lapack.SVDOverwrite or lapack.SVDNone, vt is
// not used.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgesvd but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dgesvd will panic.
//
// Dgesvd returns whether the decomposition successfully completed.
func (impl Implementation) Dgesvd(jobU, jobVT lapack.SVDJob, m, n int, a []float64, lda int, s, u []float64, ldu int, vt []float64, ldvt int, work []float64, lwork int) (ok bool) {
	checkMatrix(m, n, a, lda)
	if jobU == lapack.SVDAll {
		checkMatrix(m, m, u, ldu)
	} else if jobU == lapack.SVDInPlace {
		checkMatrix(m, min(m, n), u, ldu)
	}
	if jobVT == lapack.SVDAll {
		checkMatrix(n, n, vt, ldvt)
	} else if jobVT == lapack.SVDInPlace {
		checkMatrix(min(m, n), n, vt, ldvt)
	}
	if jobU == lapack.SVDOverwrite && jobVT == lapack.SVDOverwrite {
		panic(noSVDO)
	}
	if len(s) < min(m, n) {
		panic(badS)
	}
	if jobU == lapack.SVDOverwrite || jobVT == lapack.SVDOverwrite {
		panic("lapack: SVD not coded to overwrite original matrix")
	}
	minWork := max(5*min(m, n), 3*min(m, n)+max(m, n))
	if lwork != -1 {
		if len(work) < lwork {
			panic(badWork)
		}
		if lwork < minWork {
			panic(badWork)
		}
	}
	if lwork == -1 {
		work[0] = float64(minWork)
		return true
	}
	return lapacke.Dgesvd(lapack.Job(jobU), lapack.Job(jobVT), m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork)
}

// Dgetf2 computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of a into
//  A = P * L * U
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Dgetf2 returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
func (Implementation) Dgetf2(m, n int, a []float64, lda int, ipiv []int) (ok bool) {
	mn := min(m, n)
	checkMatrix(m, n, a, lda)
	if len(ipiv) < mn {
		panic(badIpiv)
	}
	ipiv32 := make([]int32, len(ipiv))
	ok = lapacke.Dgetf2(m, n, a, lda, ipiv32)
	for i, v := range ipiv32 {
		ipiv[i] = int(v) - 1 // Transform to zero-indexed.
	}
	return ok
}

// Dgetrf computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of A into
//  A = P * L * U
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Dgetrf is the blocked version of the algorithm.
//
// Dgetrf returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
func (impl Implementation) Dgetrf(m, n int, a []float64, lda int, ipiv []int) (ok bool) {
	mn := min(m, n)
	checkMatrix(m, n, a, lda)
	if len(ipiv) < mn {
		panic(badIpiv)
	}
	ipiv32 := make([]int32, len(ipiv))
	ok = lapacke.Dgetrf(m, n, a, lda, ipiv32)
	for i, v := range ipiv32 {
		ipiv[i] = int(v) - 1 // Transform to zero-indexed.
	}
	return ok
}

// Dgetri computes the inverse of the matrix A using the LU factorization computed
// by Dgetrf. On entry, a contains the PLU decomposition of A as computed by
// Dgetrf and on exit contains the reciprocal of the original matrix.
//
// Dtrtri will not perform the inversion if the matrix is singular, and returns
// a boolean indicating whether the inversion was successful.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgetri but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dgetri will panic.
func (impl Implementation) Dgetri(n int, a []float64, lda int, ipiv []int, work []float64, lwork int) (ok bool) {
	checkMatrix(n, n, a, lda)
	if len(ipiv) < n {
		panic(badIpiv)
	}
	if lwork == -1 {
		work[0] = float64(n)
		return true
	}
	if lwork < n {
		panic(badWork)
	}
	if len(work) < lwork {
		panic(badWork)
	}
	ipiv32 := make([]int32, len(ipiv))
	for i, v := range ipiv {
		ipiv32[i] = int32(v) + 1 // Transform to one-indexed.
	}
	return lapacke.Dgetri(n, a, lda, ipiv32, work, lwork)
}

// Dgetrs solves a system of equations using an LU factorization.
// The system of equations solved is
//  A * X = B if trans == blas.Trans
//  A^T * X = B if trans == blas.NoTrans
// A is a general n×n matrix with stride lda. B is a general matrix of size n×nrhs.
//
// On entry b contains the elements of the matrix B. On exit, b contains the
// elements of X, the solution to the system of equations.
//
// a and ipiv contain the LU factorization of A and the permutation indices as
// computed by Dgetrf. ipiv is zero-indexed.
func (impl Implementation) Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int) {
	checkMatrix(n, n, a, lda)
	checkMatrix(n, nrhs, b, ldb)
	if len(ipiv) < n {
		panic(badIpiv)
	}
	ipiv32 := make([]int32, len(ipiv))
	for i, v := range ipiv {
		ipiv32[i] = int32(v) + 1 // Transform to one-indexed.
	}
	lapacke.Dgetrs(trans, n, nrhs, a, lda, ipiv32, b, ldb)
}

// Dorgbr generates one of the matrices Q or P^T computed by Dgebrd.
// See Dgebrd for the description of Q and P^T.
//
// If vect == lapack.ApplyQ, then a is assumed to have been an m×k matrix and
// Q is of order m. If m >= k, then Dorgbr returns the first n columns of Q
// where m >= n >= k. If m < k, then Dorgbr returns Q as an m×m matrix.
//
// If vect == lapack.ApplyP, then A is assumed to have been a k×n matrix, and
// P^T is of order n. If k < n, then Dorgbr returns the first m rows of P^T,
// where n >= m >= k. If k >= n, then Dorgbr returns P^T as an n×n matrix.
func (impl Implementation) Dorgbr(vect lapack.DecompUpdate, m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	mn := min(m, n)
	wantq := vect == lapack.ApplyQ
	if wantq {
		if m < n || n < min(m, k) || m < min(m, k) {
			panic(badDims)
		}
	} else {
		if n < m || m < min(n, k) || n < min(n, k) {
			panic(badDims)
		}
	}
	if wantq {
		checkMatrix(m, k, a, lda)
	} else {
		checkMatrix(k, n, a, lda)
	}
	if lwork == -1 {
		work[0] = float64(mn)
		return
	}
	if len(work) < lwork {
		panic(badWork)
	}
	if lwork < mn {
		panic(badWork)
	}
	lapacke.Dorgbr(byte(vect), m, n, k, a, lda, tau, work, lwork)
}

// Dorghr generates an n×n orthogonal matrix Q which is defined as the product
// of ihi-ilo elementary reflectors:
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
//
// a and lda represent an n×n matrix that contains the elementary reflectors, as
// returned by Dgehrd. On return, a is overwritten by the n×n orthogonal matrix
// Q. Q will be equal to the identity matrix except in the submatrix
// Q[ilo+1:ihi+1,ilo+1:ihi+1].
//
// ilo and ihi must have the same values as in the previous call of Dgehrd. It
// must hold that
//  0 <= ilo <= ihi < n,  if n > 0,
//  ilo = 0, ihi = -1,    if n == 0.
//
// tau contains the scalar factors of the elementary reflectors, as returned by
// Dgehrd. tau must have length n-1.
//
// work must have length at least max(1,lwork) and lwork must be at least
// ihi-ilo. For optimum performance lwork must be at least (ihi-ilo)*nb where nb
// is the optimal blocksize. On return, work[0] will contain the optimal value
// of lwork.
//
// If lwork == -1, instead of performing Dorghr, only the optimal value of lwork
// will be stored into work[0].
//
// If any requirement on input sizes is not met, Dorghr will panic.
//
// Dorghr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorghr(n, ilo, ihi int, a []float64, lda int, tau, work []float64, lwork int) {
	checkMatrix(n, n, a, lda)
	nh := ihi - ilo
	switch {
	case ilo < 0 || max(1, n) <= ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lwork < max(1, nh) && lwork != -1:
		panic(badWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}
	lapacke.Dorghr(n, ilo+1, ihi+1, a, lda, tau, work, lwork)
}

// Dorglq generates an m×n matrix Q with orthonormal rows defined by the product
// of elementary reflectors
//  Q = H_{k-1} * ... * H_1 * H_0
// as computed by Dgelqf. Dorglq is the blocked version of Dorgl2 that makes
// greater use of level-3 BLAS routines.
//
// len(tau) >= k, 0 <= k <= n, and 0 <= m <= n.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dorglq but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dorglq will panic, and at minimum
// lwork >= m.
//
// Dorglq will panic if the conditions on input values are not met.
func (impl Implementation) Dorglq(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	if lwork == -1 {
		work[0] = float64(m)
		return
	}
	checkMatrix(m, n, a, lda)
	if k < 0 {
		panic(kLT0)
	}
	if k > m {
		panic(kGTM)
	}
	if m > n {
		panic(nLTM)
	}
	if len(tau) < k {
		panic(badTau)
	}
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork < m {
		panic(badWork)
	}
	lapacke.Dorglq(m, n, k, a, lda, tau, work, lwork)
}

// Dorgqr generates an m×n matrix Q with orthonormal columns defined by the
// product of elementary reflectors
//  Q = H_0 * H_1 * ... * H_{k-1}
// as computed by Dgeqrf. Dorgqr is the blocked version of Dorg2r that makes
// greater use of level-3 BLAS routines.
//
// len(tau) >= k, 0 <= k <= n, and 0 <= n <= m.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dorgqr but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dorgqr will panic, and at minimum
// lwork >= n.
//
// Dorgqr will panic if the conditions on input values are not met.
func (impl Implementation) Dorgqr(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	if lwork == -1 {
		work[0] = float64(n)
		return
	}
	checkMatrix(m, n, a, lda)
	if k < 0 {
		panic(kLT0)
	}
	if k > n {
		panic(kGTN)
	}
	if n > m {
		panic(mLTN)
	}
	if len(tau) < k {
		panic(badTau)
	}
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork < n {
		panic(badWork)
	}
	lapacke.Dorgqr(m, n, k, a, lda, tau, work, lwork)
}

// Dormbr applies a multiplicative update to the matrix C based on a
// decomposition computed by Dgebrd.
//
// Dormbr computes
//  Q * C if vect == lapack.ApplyQ, side == blas.Left, and trans == blas.NoTrans
//  C * Q if vect == lapack.ApplyQ, side == blas.Right, and trans == blas.NoTrans
//  Q^T * C if vect == lapack.ApplyQ, side == blas.Left, and trans == blas.Trans
//  C * Q^T if vect == lapack.ApplyQ, side == blas.Right, and trans == blas.Trans
//
//  P * C if vect == lapack.ApplyP, side == blas.Left, and trans == blas.NoTrans
//  C * P if vect == lapack.ApplyP, side == blas.Left, and trans == blas.NoTrans
//  P^T * C if vect == lapack.ApplyP, side == blas.Right, and trans == blas.Trans
//  C * P^T if vect == lapack.ApplyP, side == blas.Right, and trans == blas.Trans
// where P and Q are the orthogonal matrices determined by Dgebrd, A = Q * B * P^T.
// See Dgebrd for the definitions of Q and P.
//
// If vect == lapack.ApplyQ, A is assumed to have been an nq×k matrix, while if
// vect == lapack.ApplyP, A is assumed to have been a k×nq matrix. nq = m if
// side == blas.Left, while nq = n if side == blas.Right.
//
// C is an m×n matrix. On exit it is updated by the multiplication listed above.
//
// Tau must have length min(nq,k), and Dormbr will panic otherwise. Tau contains
// the elementary reflectors to construct Q or P depending on the value of
// vect.
func (impl Implementation) Dormbr(vect lapack.DecompUpdate, side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	if side != blas.Left && side != blas.Right {
		panic(badSide)
	}
	if trans != blas.NoTrans && trans != blas.Trans {
		panic(badTrans)
	}
	if vect != lapack.ApplyP && vect != lapack.ApplyQ {
		panic(badDecompUpdate)
	}
	nq := n
	if side == blas.Left {
		nq = m
	}
	if vect == lapack.ApplyQ {
		checkMatrix(nq, min(nq, k), a, lda)
	} else {
		checkMatrix(min(nq, k), nq, a, lda)
	}
	lapacke.Dormbr(byte(vect), side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork)
}

// Dormhr multiplies an m×n general matrix C with an nq×nq orthogonal matrix Q
//  Q * C,    if side == blas.Left and trans == blas.NoTrans,
//  Q^T * C,  if side == blas.Left and trans == blas.Trans,
//  C * Q,    if side == blas.Right and trans == blas.NoTrans,
//  C * Q^T,  if side == blas.Right and trans == blas.Trans,
// where nq == m if side == blas.Left and nq == n if side == blas.Right.
//
// Q is defined implicitly as the product of ihi-ilo elementary reflectors, as
// returned by Dgehrd:
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
// Q is equal to the identity matrix except in the submatrix
// Q[ilo+1:ihi+1,ilo+1:ihi+1].
//
// ilo and ihi must have the same values as in the previous call of Dgehrd. It
// must hold that
//  0 <= ilo <= ihi < m,   if m > 0 and side == blas.Left,
//  ilo = 0 and ihi = -1,  if m = 0 and side == blas.Left,
//  0 <= ilo <= ihi < n,   if n > 0 and side == blas.Right,
//  ilo = 0 and ihi = -1,  if n = 0 and side == blas.Right.
//
// a and lda represent an m×m matrix if side == blas.Left and an n×n matrix if
// side == blas.Right. The matrix contains vectors which define the elementary
// reflectors, as returned by Dgehrd.
//
// tau contains the scalar factors of the elementary reflectors, as returned by
// Dgehrd. tau must have length m-1 if side == blas.Left and n-1 if side ==
// blas.Right.
//
// c and ldc represent the m×n matrix C. On return, c is overwritten by the
// product with Q.
//
// work must have length at least max(1,lwork), and lwork must be at least
// max(1,n), if side == blas.Left, and max(1,m), if side == blas.Right. For
// optimum performance lwork should be at least n*nb if side == blas.Left and
// m*nb if side == blas.Right, where nb is the optimal block size. On return,
// work[0] will contain the optimal value of lwork.
//
// If lwork == -1, instead of performing Dormhr, only the optimal value of lwork
// will be stored in work[0].
//
// If any requirement on input sizes is not met, Dormhr will panic.
//
// Dormhr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dormhr(side blas.Side, trans blas.Transpose, m, n, ilo, ihi int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	var (
		nq int // The order of Q.
		nw int // The minimum length of work.
	)
	switch side {
	case blas.Left:
		nq = m
		nw = n
	case blas.Right:
		nq = n
		nw = m
	default:
		panic(badSide)
	}
	switch {
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case ilo < 0 || max(1, nq) <= ilo:
		panic(badIlo)
	case ihi < min(ilo, nq-1) || nq <= ihi:
		panic(badIhi)
	case lwork < max(1, nw) && lwork != -1:
		panic(badWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}
	if lwork != -1 {
		checkMatrix(m, n, c, ldc)
		checkMatrix(nq, nq, a, lda)
		if len(tau) != nq-1 && nq > 0 {
			panic(badTau)
		}
	}
	lapacke.Dormhr(side, trans, m, n, ilo+1, ihi+1, a, lda, tau, c, ldc, work, lwork)
}

// Dormlq multiplies the matrix C by the orthogonal matrix Q defined by the
// slices a and tau. A and tau are as returned from Dgelqf.
//  C = Q * C    if side == blas.Left and trans == blas.NoTrans
//  C = Q^T * C  if side == blas.Left and trans == blas.Trans
//  C = C * Q    if side == blas.Right and trans == blas.NoTrans
//  C = C * Q^T  if side == blas.Right and trans == blas.Trans
// If side == blas.Left, A is a matrix of side k×m, and if side == blas.Right
// A is of size k×n. This uses a blocked algorithm.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m if side == blas.Left and lwork >= n if side == blas.Right,
// and this function will panic otherwise.
// Dormlq uses a block algorithm, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dormlq,
// the optimal work length will be stored into work[0].
//
// tau contains the Householder scales and must have length at least k, and
// this function will panic otherwise.
func (impl Implementation) Dormlq(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	if side != blas.Left && side != blas.Right {
		panic(badSide)
	}
	if trans != blas.Trans && trans != blas.NoTrans {
		panic(badTrans)
	}
	left := side == blas.Left
	if left {
		checkMatrix(k, m, a, lda)
	} else {
		checkMatrix(k, n, a, lda)
	}
	checkMatrix(m, n, c, ldc)
	if len(tau) < k {
		panic(badTau)
	}
	if lwork == -1 {
		if left {
			work[0] = float64(n)
			return
		}
		work[0] = float64(m)
		return
	}
	if left {
		if lwork < n {
			panic(badWork)
		}
	} else {
		if lwork < m {
			panic(badWork)
		}
	}
	lapacke.Dormlq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork)
}

// Dormqr multiplies an m×n matrix C by an orthogonal matrix Q as
//  C = Q * C,    if side == blas.Left  and trans == blas.NoTrans,
//  C = Q^T * C,  if side == blas.Left  and trans == blas.Trans,
//  C = C * Q,    if side == blas.Right and trans == blas.NoTrans,
//  C = C * Q^T,  if side == blas.Right and trans == blas.Trans,
// where Q is defined as the product of k elementary reflectors
//  Q = H_0 * H_1 * ... * H_{k-1}.
//
// If side == blas.Left, A is an m×k matrix and 0 <= k <= m.
// If side == blas.Right, A is an n×k matrix and 0 <= k <= n.
// The ith column of A contains the vector which defines the elementary
// reflector H_i and tau[i] contains its scalar factor. tau must have length k
// and Dormqr will panic otherwise. Dgeqrf returns A and tau in the required
// form.
//
// work must have length at least max(1,lwork), and lwork must be at least n if
// side == blas.Left and at least m if side == blas.Right, otherwise Dormqr will
// panic.
//
// work is temporary storage, and lwork specifies the usable memory length. At
// minimum, lwork >= m if side == blas.Left and lwork >= n if side ==
// blas.Right, and this function will panic otherwise. Larger values of lwork
// will generally give better performance. On return, work[0] will contain the
// optimal value of lwork.
//
// If lwork is -1, instead of performing Dormqr, the optimal workspace size will
// be stored into work[0].
func (impl Implementation) Dormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	var nq, nw int
	switch side {
	default:
		panic(badSide)
	case blas.Left:
		nq = m
		nw = n
	case blas.Right:
		nq = n
		nw = m
	}
	switch {
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case m < 0 || n < 0:
		panic(negDimension)
	case k < 0 || nq < k:
		panic("lapack: invalid value of k")
	case len(work) < lwork:
		panic(shortWork)
	case lwork < max(1, nw) && lwork != -1:
		panic(badWork)
	}
	if lwork != -1 {
		checkMatrix(nq, k, a, lda)
		checkMatrix(m, n, c, ldc)
		if len(tau) != k {
			panic(badTau)
		}
	}

	lapacke.Dormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork)
}

// Dpocon estimates the reciprocal of the condition number of a positive-definite
// matrix A given the Cholesky decomposition of A. The condition number computed
// is based on the 1-norm and the ∞-norm.
//
// anorm is the 1-norm and the ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 3*n and Dpocon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dpocon will panic otherwise.
// Elements of iwork must fit within the int32 type or Dpocon will panic.
func (impl Implementation) Dpocon(uplo blas.Uplo, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64 {
	checkMatrix(n, n, a, lda)
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if len(work) < 3*n {
		panic(badWork)
	}
	if len(iwork) < n {
		panic(badWork)
	}
	rcond := make([]float64, 1)
	_iwork := make([]int32, len(iwork))
	for i, v := range iwork {
		if v != int(int32(v)) {
			panic("lapack: iwork element out of range")
		}
		_iwork[i] = int32(v)
	}
	lapacke.Dpocon(uplo, n, a, lda, anorm, rcond, work, _iwork)
	for i, v := range _iwork {
		iwork[i] = int(v)
	}
	return rcond[0]
}

// Dsyev computes all eigenvalues and, optionally, the eigenvectors of a real
// symmetric matrix A.
//
// w contains the eigenvalues in ascending order upon return. w must have length
// at least n, and Dsyev will panic otherwise.
//
// On entry, a contains the elements of the symmetric matrix A in the triangular
// portion specified by uplo. If jobz == lapack.ComputeEV a contains the
// orthonormal eigenvectors of A on exit, otherwise on exit the specified
// triangular region is overwritten.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dsyev but will instead write the minimum
// work necessary to work[0]. If len(work) < lwork, Dsyev will panic.
func (impl Implementation) Dsyev(jobz lapack.EVJob, uplo blas.Uplo, n int, a []float64, lda int, w, work []float64, lwork int) (ok bool) {
	checkMatrix(n, n, a, lda)
	if lwork == -1 {
		work[0] = 3*float64(n) - 1
		return
	}
	if len(work) < lwork {
		panic(badWork)
	}
	if lwork < 3*n-1 {
		panic(badWork)
	}
	return lapacke.Dsyev(lapack.Job(jobz), uplo, n, a, lda, w, work, lwork)
}

// Dtrcon estimates the reciprocal of the condition number of a triangular matrix A.
// The condition number computed may be based on the 1-norm or the ∞-norm.
//
// work is a temporary data slice of length at least 3*n and Dtrcon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dtrcon will panic otherwise.
// Elements of iwork must fit within the int32 type or Dtrcon will panic.
func (impl Implementation) Dtrcon(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int, work []float64, iwork []int) float64 {
	if norm != lapack.MaxColumnSum && norm != lapack.MaxRowSum {
		panic(badNorm)
	}
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if diag != blas.NonUnit && diag != blas.Unit {
		panic(badDiag)
	}
	if len(work) < 3*n {
		panic(badWork)
	}
	if len(iwork) < n {
		panic(badWork)
	}
	rcond := []float64{0}
	_iwork := make([]int32, len(iwork))
	for i, v := range iwork {
		if v != int(int32(v)) {
			panic("lapack: iwork element out of range")
		}
		_iwork[i] = int32(v)
	}
	lapacke.Dtrcon(byte(norm), uplo, diag, n, a, lda, rcond, work, _iwork)
	for i, v := range _iwork {
		iwork[i] = int(v)
	}
	return rcond[0]
}

// Dtrexc reorders the real Schur factorization of a n×n real matrix
//  A = Q*T*Q^T
// so that the diagonal block of T with row index ifst is moved to row ilst.
//
// On entry, T must be in Schur canonical form, that is, block upper triangular
// with 1×1 and 2×2 diagonal blocks; each 2×2 diagonal block has its diagonal
// elements equal and its off-diagonal elements of opposite sign.
//
// On return, T will be reordered by an orthogonal similarity transformation Z
// as Z^T*T*Z, and will be again in Schur canonical form.
//
// If compq is lapack.UpdateSchur, on return the matrix Q of Schur vectors will be
// updated by postmultiplying it with Z.
// If compq is lapack.None, the matrix Q is not referenced and will not be
// updated.
// For other values of compq Dtrexc will panic.
//
// ifst and ilst specify the reordering of the diagonal blocks of T. The block
// with row index ifst is moved to row ilst, by a sequence of transpositions
// between adjacent blocks.
//
// If ifst points to the second row of a 2×2 block, ifstOut will point to the
// first row, otherwise it will be equal to ifst.
//
// ilstOut will point to the first row of the block in its final position. If ok
// is true, ilstOut may differ from ilst by +1 or -1.
//
// It must hold that
//  0 <= ifst < n, and  0 <= ilst < n,
// otherwise Dtrexc will panic.
//
// If ok is false, two adjacent blocks were too close to swap because the
// problem is very ill-conditioned. T may have been partially reordered, and
// ilstOut will point to the first row of the block at the position to which it
// has been moved.
//
// work must have length at least n, otherwise Dtrexc will panic.
//
// Dtrexc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtrexc(compq lapack.EVComp, n int, t []float64, ldt int, q []float64, ldq int, ifst, ilst int, work []float64) (ifstOut, ilstOut int, ok bool) {
	checkMatrix(n, n, t, ldt)
	switch compq {
	default:
		panic("lapack: bad value of compq")
	case lapack.None:
		// q is not referenced but LAPACKE checks that ldq >= n always.
		q = nil
		ldq = max(1, n)
	case lapack.UpdateSchur:
		checkMatrix(n, n, q, ldq)
	}
	if (ifst < 0 || n <= ifst) && n > 0 {
		panic("lapack: ifst out of range")
	}
	if (ilst < 0 || n <= ilst) && n > 0 {
		panic("lapack: ilst out of range")
	}
	if len(work) < n {
		panic(badWork)
	}

	// Quick return if possible.
	if n <= 1 {
		return ifst, ilst, true
	}

	ifst32 := []int32{int32(ifst + 1)}
	ilst32 := []int32{int32(ilst + 1)}
	ok = lapacke.Dtrexc(lapack.Comp(compq), n, t, ldt, q, ldq, ifst32, ilst32, work)
	ifst = int(ifst32[0] - 1)
	ilst = int(ilst32[0] - 1)
	return ifst, ilst, ok
}

// Dtrtri computes the inverse of a triangular matrix, storing the result in place
// into a. This is the BLAS level 3 version of the algorithm which builds upon
// Dtrti2 to operate on matrix blocks instead of only individual columns.
//
// Dtrtri returns whether the matrix a is singular.
// If the matrix is singular, the inversion is not performed.
func (impl Implementation) Dtrtri(uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int) (ok bool) {
	checkMatrix(n, n, a, lda)
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if diag != blas.NonUnit && diag != blas.Unit {
		panic(badDiag)
	}
	return lapacke.Dtrtri(uplo, diag, n, a, lda)
}

// Dtrtrs solves a triangular system of the form A * X = B or A^T * X = B.
// Dtrtrs returns whether the solve completed successfully.
// If A is singular, no solve is performed.
func (impl Implementation) Dtrtrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, nrhs int, a []float64, lda int, b []float64, ldb int) (ok bool) {
	return lapacke.Dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb)
}

// Dhseqr computes the eigenvalues of an n×n Hessenberg matrix H and,
// optionally, the matrices T and Z from the Schur decomposition
//  H = Z T Z^T,
// where T is an n×n upper quasi-triangular matrix (the Schur form), and Z is
// the n×n orthogonal matrix of Schur vectors.
//
// Optionally Z may be postmultiplied into an input orthogonal matrix Q so that
// this routine can give the Schur factorization of a matrix A which has been
// reduced to the Hessenberg form H by the orthogonal matrix Q:
//  A = Q H Q^T = (QZ) T (QZ)^T.
//
// If job == lapack.EigenvaluesOnly, only the eigenvalues will be computed.
// If job == lapack.EigenvaluesAndSchur, the eigenvalues and the Schur form T will
// be computed.
// For other values of job Dhseqr will panic.
//
// If compz == lapack.None, no Schur vectors will be computed and Z will not be
// referenced.
// If compz == lapack.HessEV, on return Z will contain the matrix of Schur
// vectors of H.
// If compz == lapack.OriginalEV, on entry z is assumed to contain the orthogonal
// matrix Q that is the identity except for the submatrix
// Q[ilo:ihi+1,ilo:ihi+1]. On return z will be updated to the product Q*Z.
//
// ilo and ihi determine the block of H on which Dhseqr operates. It is assumed
// that H is already upper triangular in rows and columns [0:ilo] and [ihi+1:n],
// although it will be only checked that the block is isolated, that is,
//  ilo == 0   or H[ilo,ilo-1] == 0,
//  ihi == n-1 or H[ihi+1,ihi] == 0,
// and Dhseqr will panic otherwise. ilo and ihi are typically set by a previous
// call to Dgebal, otherwise they should be set to 0 and n-1, respectively. It
// must hold that
//  0 <= ilo <= ihi < n,     if n > 0,
//  ilo == 0 and ihi == -1,  if n == 0.
//
// wr and wi must have length n.
//
// work must have length at least lwork and lwork must be at least max(1,n)
// otherwise Dhseqr will panic. The minimum lwork delivers very good and
// sometimes optimal performance, although lwork as large as 11*n may be
// required. On return, work[0] will contain the optimal value of lwork.
//
// If lwork is -1, instead of performing Dhseqr, the function only estimates the
// optimal workspace size and stores it into work[0]. Neither h nor z are
// accessed.
//
// unconverged indicates whether Dhseqr computed all the eigenvalues.
//
// If unconverged == 0, all the eigenvalues have been computed and their real
// and imaginary parts will be stored on return in wr and wi, respectively. If
// two eigenvalues are computed as a complex conjugate pair, they are stored in
// consecutive elements of wr and wi, say the i-th and (i+1)th, with wi[i] > 0
// and wi[i+1] < 0.
//
// If unconverged == 0 and job == lapack.EigenvaluesAndSchur, on return H will
// contain the upper quasi-triangular matrix T from the Schur decomposition (the
// Schur form). 2×2 diagonal blocks (corresponding to complex conjugate pairs of
// eigenvalues) will be returned in standard form, with
//  H[i,i] == H[i+1,i+1],
// and
//  H[i+1,i]*H[i,i+1] < 0.
// The eigenvalues will be stored in wr and wi in the same order as on the
// diagonal of the Schur form returned in H, with
//  wr[i] = H[i,i],
// and, if H[i:i+2,i:i+2] is a 2×2 diagonal block,
//  wi[i]   = sqrt(-H[i+1,i]*H[i,i+1]),
//  wi[i+1] = -wi[i].
//
// If unconverged == 0 and job == lapack.EigenvaluesOnly, the contents of h
// on return is unspecified.
//
// If unconverged > 0, some eigenvalues have not converged, and the blocks
// [0:ilo] and [unconverged:n] of wr and wi will contain those eigenvalues which
// have been successfully computed. Failures are rare.
//
// If unconverged > 0 and job == lapack.EigenvaluesOnly, on return the
// remaining unconverged eigenvalues are the eigenvalues of the upper Hessenberg
// matrix H[ilo:unconverged,ilo:unconverged].
//
// If unconverged > 0 and job == lapack.EigenvaluesAndSchur, then on
// return
//  (initial H) U = U (final H),   (*)
// where U is an orthogonal matrix. The final H is upper Hessenberg and
// H[unconverged:ihi+1,unconverged:ihi+1] is upper quasi-triangular.
//
// If unconverged > 0 and compz == lapack.OriginalEV, then on return
//  (final Z) = (initial Z) U,
// where U is the orthogonal matrix in (*) regardless of the value of job.
//
// If unconverged > 0 and compz == lapack.InitZ, then on return
//  (final Z) = U,
// where U is the orthogonal matrix in (*) regardless of the value of job.
//
// References:
//  [1] R. Byers. LAPACK 3.1 xHSEQR: Tuning and Implementation Notes on the
//      Small Bulge Multi-Shift QR Algorithm with Aggressive Early Deflation.
//      LAPACK Working Note 187 (2007)
//      URL: http://www.netlib.org/lapack/lawnspdf/lawn187.pdf
//  [2] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part I:
//      Maintaining Well-Focused Shifts and Level 3 Performance. SIAM J. Matrix
//      Anal. Appl. 23(4) (2002), pp. 929—947
//      URL: http://dx.doi.org/10.1137/S0895479801384573
//  [3] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part II:
//      Aggressive Early Deflation. SIAM J. Matrix Anal. Appl. 23(4) (2002), pp. 948—973
//      URL: http://dx.doi.org/10.1137/S0895479801384585
//
// Dhseqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dhseqr(job lapack.EVJob, compz lapack.EVComp, n, ilo, ihi int, h []float64, ldh int, wr, wi []float64, z []float64, ldz int, work []float64, lwork int) (unconverged int) {
	switch job {
	default:
		panic(badEVJob)
	case lapack.EigenvaluesOnly, lapack.EigenvaluesAndSchur:
	}
	var wantz bool
	switch compz {
	default:
		panic(badEVComp)
	case lapack.None:
	case lapack.HessEV, lapack.OriginalEV:
		wantz = true
	}
	switch {
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case len(work) < lwork:
		panic(shortWork)
	case lwork < max(1, n) && lwork != -1:
		panic(badWork)
	}
	if lwork != -1 {
		checkMatrix(n, n, h, ldh)
		switch {
		case wantz:
			checkMatrix(n, n, z, ldz)
		case len(wr) < n:
			panic("lapack: wr has insufficient length")
		case len(wi) < n:
			panic("lapack: wi has insufficient length")
		}
	}

	return lapacke.Dhseqr(lapack.Job(job), lapack.Comp(compz), n, ilo+1, ihi+1,
		h, ldh, wr, wi, z, ldz, work, lwork)
}

// Dgeev computes the eigenvalues and, optionally, the left and/or right
// eigenvectors for an n×n real nonsymmetric matrix A.
//
// The right eigenvector v_j of A corresponding to an eigenvalue λ_j
// is defined by
//  A v_j = λ_j v_j,
// and the left eigenvector u_j corresponding to an eigenvalue λ_j is defined by
//  u_j^H A = λ_j u_j^H,
// where u_j^H is the conjugate transpose of u_j.
//
// On return, A will be overwritten and the left and right eigenvectors will be
// stored, respectively, in the columns of the n×n matrices VL and VR in the
// same order as their eigenvalues. If the j-th eigenvalue is real, then
//  u_j = VL[:,j],
//  v_j = VR[:,j],
// and if it is not real, then j and j+1 form a complex conjugate pair and the
// eigenvectors can be recovered as
//  u_j     = VL[:,j] + i*VL[:,j+1],
//  u_{j+1} = VL[:,j] - i*VL[:,j+1],
//  v_j     = VR[:,j] + i*VR[:,j+1],
//  v_{j+1} = VR[:,j] - i*VR[:,j+1].
// where i is the imaginary unit. The computed eigenvectors are normalized to
// have Euclidean norm equal to 1 and largest component real.
//
// Left eigenvectors will be computed only if jobvl == lapack.ComputeLeftEV,
// otherwise jobvl must be lapack.None. Right eigenvectors will be computed
// only if jobvr == lapack.ComputeRightEV, otherwise jobvr must be lapack.None.
// For other values of jobvl and jobvr Dgeev will panic.
//
// wr and wi contain the real and imaginary parts, respectively, of the computed
// eigenvalues. Complex conjugate pairs of eigenvalues appear consecutively with
// the eigenvalue having the positive imaginary part first.
// wr and wi must have length n, and Dgeev will panic otherwise.
//
// work must have length at least lwork and lwork must be at least max(1,4*n) if
// the left or right eigenvectors are computed, and at least max(1,3*n) if no
// eigenvectors are computed. For good performance, lwork must generally be
// larger.  On return, optimal value of lwork will be stored in work[0].
//
// If lwork == -1, instead of performing Dgeev, the function only calculates the
// optimal vaule of lwork and stores it into work[0].
//
// On return, first is the index of the first valid eigenvalue. If first == 0,
// all eigenvalues and eigenvectors have been computed. If first is positive,
// Dgeev failed to compute all the eigenvalues, no eigenvectors have been
// computed and wr[first:] and wi[first:] contain those eigenvalues which have
// converged.
func (impl Implementation) Dgeev(jobvl lapack.LeftEVJob, jobvr lapack.RightEVJob, n int, a []float64, lda int, wr, wi []float64, vl []float64, ldvl int, vr []float64, ldvr int, work []float64, lwork int) (first int) {
	var wantvl bool
	switch jobvl {
	default:
		panic("lapack: invalid LeftEVJob")
	case lapack.ComputeLeftEV:
		wantvl = true
	case lapack.None:
		wantvl = false
	}
	var wantvr bool
	switch jobvr {
	default:
		panic("lapack: invalid RightEVJob")
	case lapack.ComputeRightEV:
		wantvr = true
	case lapack.None:
		wantvr = false
	}
	switch {
	case n < 0:
		panic(nLT0)
	case len(work) < lwork:
		panic(shortWork)
	}
	var minwrk int
	if wantvl || wantvr {
		minwrk = max(1, 4*n)
	} else {
		minwrk = max(1, 3*n)
	}
	if lwork != -1 {
		checkMatrix(n, n, a, lda)
		if wantvl {
			checkMatrix(n, n, vl, ldvl)
		}
		if wantvr {
			checkMatrix(n, n, vr, ldvr)
		}
		switch {
		case len(wr) != n:
			panic("lapack: bad length of wr")
		case len(wi) != n:
			panic("lapack: bad length of wi")
		case lwork < minwrk:
			panic(badWork)
		}
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0
	}

	first = lapacke.Dgeev(lapack.Job(jobvl), lapack.Job(jobvr), n, a, max(n, lda), wr, wi,
		vl, max(n, ldvl), vr, max(n, ldvr), work, lwork)
	if lwork == -1 && int(work[0]) < minwrk {
		work[0] = float64(minwrk)
	}
	return first
}
