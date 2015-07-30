// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lapack64 provides a set of convenient wrapper functions for LAPACK
// calls, as specified in the netlib standard (www.netlib.org).
//
// The native Go routines are used by default, and the Use function can be used
// to set an alternate implementation.
//
// If the type of matrix (General, Symmetric, etc.) is known and fixed, it is
// used in the wrapper signature. In many cases, however, the type of the matrix
// changes during the call to the routine, for example the matrix is symmetric on
// entry and is triangular on exit. In these cases the correct types should be checked
// in the documentation.
//
// The full set of Lapack functions is very large, and it is not clear that a
// full implementation is desirable, let alone feasible. Please open up an issue
// if there is a specific function you need and/or are willing to implement.
package lapack64

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/lapack"
	"github.com/gonum/lapack/native"
)

var lapack64 lapack.Float64 = native.Implementation{}

// Use sets the LAPACK float64 implementation to be used by subsequent BLAS calls.
// The default implementation is native.Implementation.
func Use(l lapack.Float64) {
	lapack64 = l
}

// Potrf computes the cholesky factorization of a.
//  A = U^T * U if ul == blas.Upper
//  A = L * L^T if ul == blas.Lower
// The underlying data between the input matrix and output matrix is shared.
func Potrf(a blas64.Symmetric) (t blas64.Triangular, ok bool) {
	ok = lapack64.Dpotrf(a.Uplo, a.N, a.Data, a.Stride)
	t.Uplo = a.Uplo
	t.N = a.N
	t.Data = a.Data
	t.Stride = a.Stride
	t.Diag = blas.NonUnit
	return
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
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(m,n) + max(m,n,nrhs), and this function will panic
// otherwise. A longer work will enable blocked algorithms to be called.
// In the special case that lwork == -1, work[0] will be set to the optimal working
// length.
func Gels(trans blas.Transpose, a blas64.General, b blas64.General, work []float64, lwork int) bool {
	return lapack64.Dgels(trans, a.Rows, a.Cols, b.Cols, a.Data, a.Stride, b.Data, b.Stride, work, lwork)
}
