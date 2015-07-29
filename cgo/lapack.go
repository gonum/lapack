// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cgo provides an interface to bindings for a C LAPACK library.
package cgo

import (
	"github.com/gonum/blas"
	"github.com/gonum/lapack"
	"github.com/gonum/lapack/cgo/clapack"
)

// Copied from lapack/native. Keep in sync.
const (
	badDirect     = "lapack: bad direct"
	badLdA        = "lapack: index of a out of range"
	badSide       = "lapack: bad side"
	badStore      = "lapack: bad store"
	badTau        = "lapack: tau has insufficient length"
	badTrans      = "lapack: bad trans"
	badUplo       = "lapack: illegal triangle"
	badWork       = "lapack: insufficient working memory"
	badWorkStride = "lapack: insufficient working array stride"
	negDimension  = "lapack: negative matrix dimension"
	nLT0          = "lapack: n < 0"
	shortWork     = "lapack: working array shorter than declared"
)

// checkMatrix verifies the parameters of a matrix input.
// Copied from lapack/native. Keep in sync.
func checkMatrix(m, n int, a []float64, lda int) {
	if m < 0 {
		panic("lapack: has negative number of rows")
	}
	if m < 0 {
		panic("lapack: has negative number of columns")
	}
	if lda < n {
		panic("lapack: stride less than number of columns")
	}
	if len(a) < (m-1)*lda+n {
		panic("lapack: insufficient matrix slice length")
	}
}

func min(m, n int) int {
	if m < n {
		return m
	}
	return n
}

// Implementation is the cgo-based C implementation of LAPACK routines.
type Implementation struct{}

var _ lapack.Float64 = Implementation{}

// Dpotrf computes the cholesky decomposition of the symmetric positive definite
// matrix a. If ul == blas.Upper, then a is stored as an upper-triangular matrix,
// and a = U U^T is stored in place into a. If ul == blas.Lower, then a = L L^T
// is computed and stored in-place into a. If a is not positive definite, false
// is returned. This is the blocked version of the algorithm.
func (impl Implementation) Dpotrf(ul blas.Uplo, n int, a []float64, lda int) (ok bool) {
	// ul is checked in clapack.Dpotrf.
	if n < 0 {
		panic(nLT0)
	}
	if lda < n {
		panic(badLdA)
	}
	if n == 0 {
		return true
	}
	return clapack.Dpotrf(ul, n, a, lda)
}

// Gels finds a minimum-norm solution based on the matrices a and b using the
// QR or LQ factorization. Dgels returns false if the matrix
// A is singular, and true if this solution was successfully found.
//
// The minimization problem solved depends on the input parameters.
//
//  1. If m >= n and trans == blas.NoTrans, Dgels finds X such that || A*X - B||_2
//  is minimized.
//  2. If m < n and trans == blas.NoTrans, Dgels finds the minimum norm solution of
//  A * X = B.
//  3. If m >= n and trans == blas.Trans, Dgels finds the minimum norm solution of
//  A^T * X = B.
//  4. If m < n and trans == blas.Trans, Dgels finds X such that || A*X - B||_2
//  is minimized.
// Note that the least-squares solutions (cases 1 and 3) perform the minimization
// per column of B. This is not the same as finding the minimum-norm matrix.
//
// The matrix a is a general matrix of size m×n and is modified during this call.
// The input matrix b is of size max(m,n)×nrhs, and serves two purposes. On entry,
// the elements of b specify the input matrix B. B has size m×nrhs if
// trans == blas.NoTrans, and n×nrhs if trans == blas.Trans. On exit, the
// leading submatrix of b contains the solution vectors X. If trans == blas.NoTrans,
// this submatrix is of size n×nrhs, and of size m×nrhs otherwise.
//
// The C interface does not support providing temporary storage. To provide compatibility
// with native, lwork == -1 will not run Dgels but will instead write to work[0].
// If len(work) < lwork, Dgels will panic.
func (impl Implementation) Dgels(trans blas.Transpose, m, n, nrhs int, a []float64, lda int, b []float64, ldb int, work []float64, lwork int) bool {
	checkMatrix(m, n, a, lda)
	mn := min(m, n)
	checkMatrix(mn, nrhs, b, ldb)
	if lwork == -1 {
		work[0] = 1 // Make work at least 1 so repeated calls with lwork == -1 can still write to work[0].
		return true
	}
	if lwork < len(work) {
		panic(shortWork)
	}
	return clapack.Dgels(trans, m, n, nrhs, a, lda, b, ldb)
}
