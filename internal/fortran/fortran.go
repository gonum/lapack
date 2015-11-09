// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fortran provides utilities to help interface to fortran routines.
//
// The utility functions provided in fortran are not intended to be used generally
// and provide very limited friendliness. They are intended to be used in development
// of Go ports of FORTRAN routines.
package fortran

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

// TransCharacter returns the FORTRAN character for the given transpose operation.
func TransCharacter(trans blas.Transpose) byte {
	switch trans {
	case blas.NoTrans:
		return 'N'
	case blas.Trans:
		return 'T'
	case blas.ConjTrans:
		return 'C'
	default:
		panic("fortran: bad BLAS trans")
	}
}

// UploCharacter returns the FORTRAN character for the given uplo state.
func UploCharacter(uplo blas.Uplo) byte {
	switch uplo {
	case blas.Upper:
		return 'U'
	case blas.Lower:
		return 'L'
	case blas.All:
		return 'A'
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// DiagCharacter returns the FORTRAN character for the given diagonal state.
func DiagCharacter(diag blas.Diag) byte {
	switch diag {
	case blas.Unit:
		return 'U'
	case blas.NonUnit:
		return 'N'
	default:
		panic("fortran: bad BLAS diag")
	}
}

// SideCharacter returns the FORTRAN character for the given side state.
func SideCharacter(side blas.Side) byte {
	switch side {
	case blas.Left:
		return 'L'
	case blas.Right:
		return 'R'
	default:
		panic("fortran: bad BLAS side")
	}
}

// General is a column major general matrix.
type General blas64.General

// NewColMajorGeneralFrom returns a column major general matrix
// with the same dimensions and data elements as the row major a.
func NewColMajorGeneralFrom(a blas64.General) General {
	t := General{
		Rows:   a.Rows,
		Cols:   a.Cols,
		Stride: a.Rows,
		Data:   make([]float64, a.Rows*a.Cols),
	}
	t.From(a)
	return t
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions as a and have adequate backing
// data storage.
func (t General) From(a blas64.General) {
	if t.Rows != a.Rows || t.Cols != a.Cols {
		panic("fortran: mismatched dimension")
	}
	if len(t.Data) < (t.Cols-1)*t.Stride+t.Rows {
		panic("fortran: short data slice")
	}
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
		}
	}
}

// NewRowMajorGeneralFrom returns a row major general matrix
// with the same dimensions and data elements as a column major a.
func NewRowMajorGeneralFrom(a General) blas64.General {
	t := blas64.General{
		Rows:   a.Rows,
		Cols:   a.Cols,
		Stride: a.Cols,
		Data:   make([]float64, a.Rows*a.Cols),
	}
	a.To(t)
	return t
}

// To fills t with elements from the receiver. The blas64.General
// must have the same dimensions as a and have adequate backing
// data storage.
func (a General) To(t blas64.General) {
	if t.Rows != a.Rows || t.Cols != a.Cols {
		panic("fortran: mismatched dimension")
	}
	if len(t.Data) < (t.Rows-1)*t.Stride+t.Cols {
		panic("fortran: short data slice")
	}
	for i := 0; i < a.Cols; i++ {
		for j := 0; j < a.Rows; j++ {
			t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
		}
	}
}

// CopyGeneralRowMajor performs a copy of src to dst which are
// row major matrices. The dimensions of src and dst must match
// and dst must have adequate data storage, otherwise CopyGeneralRowMajor
// will panic.
func CopyGeneralRowMajor(dst, src blas64.General) {
	if dst.Rows != src.Rows || dst.Cols != src.Cols {
		panic("fortran: mismatched dimension")
	}
	if len(dst.Data) < (dst.Rows-1)*dst.Stride+dst.Cols {
		panic("fortran: short data slice")
	}
	for i := 0; i < src.Rows; i++ {
		for j := 0; j < src.Cols; j++ {
			dst.Data[i*dst.Stride+j] = src.Data[i*src.Stride+j]
		}
	}
}

// CopyGeneralColMajor performs a copy of src to dst which are
// column major matrices. The dimensions of src and dst must match
// and dst must have adequate data storage, otherwise CopyGeneralRowMajor
// will panic.
func CopyGeneralColMajor(dst, src General) {
	if dst.Rows != src.Rows || dst.Cols != src.Cols {
		panic("fortran: mismatched dimension")
	}
	if len(dst.Data) < (dst.Cols-1)*dst.Stride+dst.Rows {
		panic("fortran: short data slice")
	}
	for j := 0; j < src.Cols; j++ {
		for i := 0; i < src.Rows; i++ {
			dst.Data[i+j*dst.Stride] = src.Data[i+j*src.Stride]
		}
	}
}

// Symmetric is a column major symmetric matrix.
type Symmetric blas64.Symmetric

// NewColMajorSymmetricFrom returns a column major symmetric matrix
// with the same dimensions and data elements as the row major a.
func NewColMajorSymmetricFrom(a blas64.Symmetric) Symmetric {
	t := Symmetric{
		N:      a.N,
		Stride: a.N,
		Data:   make([]float64, a.N*a.N),
		Uplo:   a.Uplo,
	}
	t.From(a)
	return t
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions as a and have adequate backing
// data storage.
func (t Symmetric) From(a blas64.Symmetric) {
	switch a.Uplo {
	case blas.Upper:
		for i := 0; i < a.N; i++ {
			for j := i; j < a.N; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	case blas.Lower:
		for i := 0; i < a.N; i++ {
			for j := 0; j <= i; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// NewRowMajorSymmetricFrom returns a row major symmetric matrix
// with the same dimensions and data elements as the col major a.
func NewRowMajorSymmetricFrom(a Symmetric) blas64.Symmetric {
	t := blas64.Symmetric{
		N:      a.N,
		Stride: a.N,
		Data:   make([]float64, a.N*a.N),
		Uplo:   a.Uplo,
	}
	a.To(t)
	return t
}

// To fills t with elements from the receiver. The blas64.Symmetric
// must have the same dimensions as a and have adequate backing
// data storage.
func (a Symmetric) To(t blas64.Symmetric) {
	switch a.Uplo {
	case blas.Upper:
		for i := 0; i < a.N; i++ {
			for j := i; j < a.N; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	case blas.Lower:
		for i := 0; i < a.N; i++ {
			for j := 0; j <= i; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// CopySymmetricRowMajor performs a copy of src to dst which are
// row major matrices. The dimensions and shape of src and dst must match
// and dst must have adequate data storage, otherwise CopyTriangularRowMajor
// will panic.
func CopySymmetricRowMajor(dst, src blas64.Symmetric) {
	if dst.N != src.N {
		panic("fortran: mismatched dimension")
	}
	if dst.Uplo != src.Uplo {
		panic("fortran: mismatched BLAS uplo")
	}
	if len(dst.Data) < (dst.N-1)*dst.Stride+dst.N {
		panic("fortran: short data slice")
	}
	switch src.Uplo {
	case blas.Upper:
		for i := 0; i < src.N; i++ {
			for j := i; j < src.N; j++ {
				dst.Data[i*dst.Stride+j] = src.Data[i*src.Stride+j]
			}
		}
	case blas.Lower:
		for i := 0; i < src.N; i++ {
			for j := 0; j <= i; j++ {
				dst.Data[i*dst.Stride+j] = src.Data[i*src.Stride+j]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// CopySymmetricColMajor performs a copy of src to dst which are
// column major matrices. The dimensions and shape of src and dst must match
// and dst must have adequate data storage, otherwise CopySymmetricColMajor
// will panic.
func CopySymmetricColMajor(dst, src Symmetric) {
	if dst.N != src.N {
		panic("fortran: mismatched dimension")
	}
	if dst.Uplo != src.Uplo {
		panic("fortran: mismatched BLAS uplo")
	}
	if len(dst.Data) < (dst.N-1)*dst.Stride+dst.N {
		panic("fortran: short data slice")
	}
	switch src.Uplo {
	case blas.Upper:
		for i := 0; i < src.N; i++ {
			for j := i; j < src.N; j++ {
				dst.Data[i+j*dst.Stride] = src.Data[i+j*src.Stride]
			}
		}
	case blas.Lower:
		for i := 0; i < src.N; i++ {
			for j := 0; j <= i; j++ {
				dst.Data[i+j*dst.Stride] = src.Data[i+j*src.Stride]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// Triangular is a column major triangular matrix.
type Triangular blas64.Triangular

// NewColMajorTriangularFrom returns a column major general matrix
// with the same dimensions and data elements as the row major a.
func NewColMajorTriangularFrom(a blas64.Triangular) Triangular {
	t := Triangular{
		N:      a.N,
		Stride: a.N,
		Data:   make([]float64, a.N*a.N),
		Diag:   a.Diag,
		Uplo:   a.Uplo,
	}
	t.From(a)
	return t
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions as a and have adequate backing
// data storage.
func (t Triangular) From(a blas64.Triangular) {
	switch a.Uplo {
	case blas.Upper:
		for i := 0; i < a.N; i++ {
			for j := i; j < a.N; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	case blas.Lower:
		for i := 0; i < a.N; i++ {
			for j := 0; j <= i; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	case blas.All:
		for i := 0; i < a.N; i++ {
			for j := 0; j < a.N; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// NewRowMajorTriangularFrom returns a row major triangular matrix
// with the same dimensions and data elements as a column major a.
func NewRowMajorTriangularFrom(a Triangular) blas64.Triangular {
	t := blas64.Triangular{
		N:      a.N,
		Stride: a.N,
		Data:   make([]float64, a.N*a.N),
		Diag:   a.Diag,
		Uplo:   a.Uplo,
	}
	a.To(t)
	return t
}

// To fills t with elements from the receiver. The blas64.Triangular
// must have the same dimensions as a and have adequate backing
// data storage.
func (a Triangular) To(t blas64.Triangular) {
	switch a.Uplo {
	case blas.Upper:
		for i := 0; i < a.N; i++ {
			for j := i; j < a.N; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	case blas.Lower:
		for i := 0; i < a.N; i++ {
			for j := 0; j <= i; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	case blas.All:
		for i := 0; i < a.N; i++ {
			for j := 0; j < a.N; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// CopyTriangularRowMajor performs a copy of src to dst which are
// row major matrices. The dimensions and shape of src and dst must match
// and dst must have adequate data storage, otherwise CopyTriangularRowMajor
// will panic. The value of src.Diag is checked for matching with dst.Diag,
// but does not alter the behavior of the copy; the underlying data is
// always copied.
func CopyTriangularRowMajor(dst, src blas64.Triangular) {
	if dst.N != src.N {
		panic("fortran: mismatched dimension")
	}
	if dst.Diag != src.Diag {
		panic("fortran: mismatched BLAS diag")
	}
	if dst.Uplo != src.Uplo {
		panic("fortran: mismatched BLAS uplo")
	}
	if len(dst.Data) < (dst.N-1)*dst.Stride+dst.N {
		panic("fortran: short data slice")
	}
	switch src.Uplo {
	case blas.Upper:
		for i := 0; i < src.N; i++ {
			for j := i; j < src.N; j++ {
				dst.Data[i*dst.Stride+j] = src.Data[i*src.Stride+j]
			}
		}
	case blas.Lower:
		for i := 0; i < src.N; i++ {
			for j := 0; j <= i; j++ {
				dst.Data[i*dst.Stride+j] = src.Data[i*src.Stride+j]
			}
		}
	case blas.All:
		for i := 0; i < src.N; i++ {
			for j := 0; j < src.N; j++ {
				dst.Data[i*dst.Stride+j] = src.Data[i*src.Stride+j]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}

// CopyTriangularColMajor performs a copy of src to dst which are
// column major matrices. The dimensions and shape of src and dst must match
// and dst must have adequate data storage, otherwise CopyTriangularColMajor
// will panic. The value of src.Diag is checked for matching with dst.Diag,
// but does not alter the behavior of the copy; the underlying data is
// always copied.
func CopyTriangularColMajor(dst, src Triangular) {
	if dst.N != src.N {
		panic("fortran: mismatched dimension")
	}
	if dst.Diag != src.Diag {
		panic("fortran: mismatched BLAS diag")
	}
	if dst.Uplo != src.Uplo {
		panic("fortran: mismatched BLAS uplo")
	}
	if len(dst.Data) < (dst.N-1)*dst.Stride+dst.N {
		panic("fortran: short data slice")
	}
	switch src.Uplo {
	case blas.Upper:
		for i := 0; i < src.N; i++ {
			for j := i; j < src.N; j++ {
				dst.Data[i+j*dst.Stride] = src.Data[i+j*src.Stride]
			}
		}
	case blas.Lower:
		for i := 0; i < src.N; i++ {
			for j := 0; j <= i; j++ {
				dst.Data[i+j*dst.Stride] = src.Data[i+j*src.Stride]
			}
		}
	case blas.All:
		for i := 0; i < src.N; i++ {
			for j := 0; j < src.N; j++ {
				dst.Data[i+j*dst.Stride] = src.Data[i+j*src.Stride]
			}
		}
	default:
		panic("fortran: bad BLAS uplo")
	}
}
