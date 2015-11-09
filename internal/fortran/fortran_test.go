// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fortran

import (
	"math"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

type general interface {
	dims() (r, c int)
	at(r, c int) float64
}

type rowMajorGeneral blas64.General

func (m rowMajorGeneral) dims() (r, c int)    { return m.Rows, m.Cols }
func (m rowMajorGeneral) at(r, c int) float64 { return m.Data[r*m.Stride+c] }

type colMajorGeneral blas64.General

func (m colMajorGeneral) dims() (r, c int)    { return m.Rows, m.Cols }
func (m colMajorGeneral) at(r, c int) float64 { return m.Data[r+c*m.Stride] }

func equalGeneral(a, b general) bool {
	ar, ac := a.dims()
	br, bc := b.dims()
	if ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if a.at(i, j) != b.at(i, j) || math.IsNaN(a.at(i, j)) != math.IsNaN(b.at(i, j)) {
				return false
			}
		}
	}
	return true
}

var generalTests = []blas64.General{
	{Rows: 2, Cols: 3, Stride: 3, Data: []float64{
		1, 2, 3,
		4, 5, 6,
	}},
	{Rows: 3, Cols: 2, Stride: 2, Data: []float64{
		1, 2,
		3, 4,
		5, 6,
	}},
	{Rows: 3, Cols: 3, Stride: 3, Data: []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}},
	{Rows: 2, Cols: 3, Stride: 5, Data: []float64{
		1, 2, 3, 0, 0,
		4, 5, 6, 0, 0,
	}},
	{Rows: 3, Cols: 2, Stride: 5, Data: []float64{
		1, 2, 0, 0, 0,
		3, 4, 0, 0, 0,
		5, 6, 0, 0, 0,
	}},
	{Rows: 3, Cols: 3, Stride: 5, Data: []float64{
		1, 2, 3, 0, 0,
		4, 5, 6, 0, 0,
		7, 8, 9, 0, 0,
	}},
}

func TestConvertGeneral(t *testing.T) {
	for _, test := range generalTests {
		colmajor := NewColMajorGeneralFrom(test)
		if !equalGeneral(colMajorGeneral(colmajor), rowMajorGeneral(test)) {
			t.Errorf("unexpected result for row major to col major conversion:\n\tgot: %+v\n\tfrom:%+v",
				colmajor, test)
		}
		rowmajor := NewRowMajorGeneralFrom(colmajor)
		if !equalGeneral(rowMajorGeneral(rowmajor), rowMajorGeneral(test)) {
			t.Errorf("unexpected result for row major to col major conversion:\n\tgot: %+v\n\twant:%+v",
				rowmajor, test)
		}
	}
}

func TestCopyGeneralRowMajor(t *testing.T) {
	for _, test := range generalTests {
		src := test
		for stride := src.Cols; stride <= src.Stride+1; stride++ {
			dst := blas64.General{
				Rows:   src.Rows,
				Cols:   src.Cols,
				Stride: stride,
				Data:   make([]float64, src.Rows*stride),
			}
			for i := range dst.Data {
				dst.Data[i] = math.NaN()
			}
			CopyGeneralRowMajor(dst, src)
			if !equalGeneral(rowMajorGeneral(dst), rowMajorGeneral(src)) {
				t.Errorf("unexpected result for row major copy:\n\tgot: %+v\n\tfrom:%+v",
					dst, src)
			}
			// We are abusing the indexing to see outside the real data.
			for i := 0; i < dst.Rows; i++ {
				for j := dst.Cols; j < dst.Stride; j++ {
					if !math.IsNaN(rowMajorGeneral(dst).at(i, j)) {
						t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
							i, j, dst, src)
					}
				}
			}
		}
	}
}

func TestCopyGeneralColMajor(t *testing.T) {
	for _, test := range generalTests {
		src := NewColMajorGeneralFrom(test)
		for stride := src.Rows; stride <= src.Stride+1; stride++ {
			dst := General{
				Rows:   src.Rows,
				Cols:   src.Cols,
				Stride: stride,
				Data:   make([]float64, src.Cols*stride),
			}
			for i := range dst.Data {
				dst.Data[i] = math.NaN()
			}
			CopyGeneralColMajor(dst, src)
			if !equalGeneral(colMajorGeneral(dst), colMajorGeneral(src)) {
				t.Errorf("unexpected result for col major copy:\n\tgot: %+v\n\tfrom:%+v",
					dst, src)
			}
			// We are abusing the indexing to see outside the real data.
			for i := dst.Rows; i < dst.Stride; i++ {
				for j := 0; j < dst.Cols; j++ {
					if !math.IsNaN(colMajorGeneral(dst).at(i, j)) {
						t.Errorf("unexpected result for col major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
							i, j, dst, src)
					}
				}
			}
		}
	}
}

type symmetric interface {
	n() int
	at(r, c int) float64
	uplo() blas.Uplo
}

type rowMajorSymmetric blas64.Symmetric

func (m rowMajorSymmetric) n() int { return m.N }
func (m rowMajorSymmetric) at(r, c int) float64 {
	if m.Uplo == blas.Lower && r < c && c < m.N {
		r, c = c, r
	}
	if m.Uplo == blas.Upper && r > c {
		r, c = c, r
	}
	return m.Data[r*m.Stride+c]
}

// atOK is at, but always returns the actual stored value and
// a boolean indicating whether it is a valid element.
func (m rowMajorSymmetric) atOK(r, c int) (v float64, ok bool) {
	ok = true
	if m.Uplo == blas.Lower && r < c && c < m.N {
		ok = false
	}
	if m.Uplo == blas.Upper && r > c {
		ok = false
	}
	return m.Data[r*m.Stride+c], ok
}
func (m rowMajorSymmetric) uplo() blas.Uplo { return m.Uplo }

type colMajorSymmetric blas64.Symmetric

func (m colMajorSymmetric) n() int { return m.N }
func (m colMajorSymmetric) at(r, c int) float64 {
	if m.Uplo == blas.Lower && r < c {
		r, c = c, r
	}
	if m.Uplo == blas.Upper && r > c && r < m.N {
		r, c = c, r
	}
	return m.Data[r+c*m.Stride]
}

// atOK is at, but always returns the actual stored value and
// a boolean indicating whether it is a valid element.
func (m colMajorSymmetric) atOK(r, c int) (v float64, ok bool) {
	ok = true
	if m.Uplo == blas.Lower && r < c {
		ok = false
	}
	if m.Uplo == blas.Upper && r > c && r < m.N {
		ok = false
	}
	return m.Data[r+c*m.Stride], ok
}
func (m colMajorSymmetric) uplo() blas.Uplo { return m.Uplo }

func equalSymmetric(a, b symmetric) bool {
	an := a.n()
	bn := b.n()
	if an != bn {
		return false
	}
	for i := 0; i < an; i++ {
		for j := 0; j < an; j++ {
			if a.at(i, j) != b.at(i, j) || math.IsNaN(a.at(i, j)) != math.IsNaN(b.at(i, j)) {
				return false
			}
		}
	}
	return true
}

var symmetricTests = []blas64.Symmetric{
	{N: 3, Stride: 3, Data: []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}},
	{N: 3, Stride: 5, Data: []float64{
		1, 2, 3, 0, 0,
		4, 5, 6, 0, 0,
		7, 8, 9, 0, 0,
	}},
}

func TestConvertSymmetric(t *testing.T) {
	for _, test := range symmetricTests {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			test.Uplo = uplo
			colmajor := NewColMajorSymmetricFrom(test)
			if !equalSymmetric(colMajorSymmetric(colmajor), rowMajorSymmetric(test)) {
				t.Errorf("unexpected result for row major to col major conversion:\n\tgot: %+v\n\tfrom:%+v",
					colmajor, test)
			}
			rowmajor := NewRowMajorSymmetricFrom(colmajor)
			if !equalSymmetric(rowMajorSymmetric(rowmajor), rowMajorSymmetric(test)) {
				t.Errorf("unexpected result for row major to col major conversion:\n\tgot: %+v\n\twant:%+v",
					rowmajor, test)
			}
		}
	}
}

func TestCopySymmetricRowMajor(t *testing.T) {
	for _, test := range symmetricTests {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			src := test
			src.Uplo = uplo
			for stride := src.N; stride <= src.Stride+1; stride++ {
				dst := blas64.Symmetric{
					N:      src.N,
					Stride: stride,
					Data:   make([]float64, src.N*stride),
					Uplo:   src.Uplo,
				}
				for i := range dst.Data {
					dst.Data[i] = math.NaN()
				}
				CopySymmetricRowMajor(dst, src)
				if !equalSymmetric(rowMajorSymmetric(dst), rowMajorSymmetric(src)) {
					t.Errorf("unexpected result for row major copy:\n\tgot: %+v\n\tfrom:%+v",
						dst, src)
				}
				// We are abusing the indexing to see outside the real data.
				for i := 0; i < dst.N; i++ {
					for j := dst.N; j < dst.Stride; j++ {
						if !math.IsNaN(rowMajorSymmetric(dst).at(i, j)) {
							t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
								i, j, dst, src)
						}
					}
				}

				// Now check the invalid triangle for overwrites.
				for i := 0; i < dst.N; i++ {
					for j := 0; j < dst.N; j++ {
						v, ok := rowMajorSymmetric(dst).atOK(i, j)
						if !ok && !math.IsNaN(v) {
							t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
								i, j, dst, src)
						}
					}
				}
			}
		}
	}
}

func TestCopySymmetricColMajor(t *testing.T) {
	for _, test := range symmetricTests {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			test.Uplo = uplo
			src := NewColMajorSymmetricFrom(test)
			for stride := src.N; stride <= src.Stride+1; stride++ {
				dst := Symmetric{
					N:      src.N,
					Stride: stride,
					Data:   make([]float64, src.N*stride),
					Uplo:   src.Uplo,
				}
				for i := range dst.Data {
					dst.Data[i] = math.NaN()
				}
				CopySymmetricColMajor(dst, src)
				if !equalSymmetric(colMajorSymmetric(dst), colMajorSymmetric(src)) {
					t.Errorf("unexpected result for col major copy:\n\tgot: %+v\n\tfrom:%+v",
						dst, src)
				}
				// We are abusing the indexing to see outside the real data.
				for i := dst.N; i < dst.Stride; i++ {
					for j := 0; j < dst.N; j++ {
						if !math.IsNaN(colMajorSymmetric(dst).at(i, j)) {
							t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
								i, j, dst, src)
						}
					}
				}

				// Now check the invalid triangle for overwrites.
				for i := 0; i < dst.N; i++ {
					for j := 0; j < dst.N; j++ {
						v, ok := colMajorSymmetric(dst).atOK(i, j)
						if !ok && !math.IsNaN(v) {
							t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
								i, j, dst, src)
						}
					}
				}
			}
		}
	}
}

type triangular interface {
	n() int
	at(r, c int) float64
	uplo() blas.Uplo
	diag() blas.Diag
}

type rowMajorTriangular blas64.Triangular

func (m rowMajorTriangular) n() int { return m.N }
func (m rowMajorTriangular) at(r, c int) float64 {
	if m.Diag == blas.Unit && r == c {
		return 1
	}
	if m.Uplo == blas.Lower && r < c && c < m.N {
		return 0
	}
	if m.Uplo == blas.Upper && r > c {
		return 0
	}
	return m.Data[r*m.Stride+c]
}

// atOK is at, but always returns the actual stored value and
// a boolean indicating whether it is a valid element.
func (m rowMajorTriangular) atOK(r, c int) (v float64, ok bool) {
	ok = true
	if m.Diag == blas.Unit && r == c {
		ok = true
	}
	if m.Uplo == blas.Lower && r < c && c < m.N {
		ok = false
	}
	if m.Uplo == blas.Upper && r > c {
		ok = false
	}
	return m.Data[r*m.Stride+c], ok
}
func (m rowMajorTriangular) uplo() blas.Uplo { return m.Uplo }
func (m rowMajorTriangular) diag() blas.Diag { return m.Diag }

type colMajorTriangular blas64.Triangular

func (m colMajorTriangular) n() int { return m.N }
func (m colMajorTriangular) at(r, c int) float64 {
	if m.Diag == blas.Unit && r == c {
		return 1
	}
	if m.Uplo == blas.Lower && r < c {
		return 0
	}
	if m.Uplo == blas.Upper && r > c && r < m.N {
		return 0
	}
	return m.Data[r+c*m.Stride]
}

// atOK is at, but always returns the actual stored value and
// a boolean indicating whether it is a valid element.
func (m colMajorTriangular) atOK(r, c int) (v float64, ok bool) {
	ok = true
	if m.Diag == blas.Unit && r == c {
		ok = true
	}
	if m.Uplo == blas.Lower && r < c {
		ok = false
	}
	if m.Uplo == blas.Upper && r > c && r < m.N {
		ok = false
	}
	return m.Data[r+c*m.Stride], ok
}
func (m colMajorTriangular) uplo() blas.Uplo { return m.Uplo }
func (m colMajorTriangular) diag() blas.Diag { return m.Diag }

func equalTriangular(a, b triangular) bool {
	an := a.n()
	bn := b.n()
	if an != bn {
		return false
	}
	for i := 0; i < an; i++ {
		for j := 0; j < an; j++ {
			if a.at(i, j) != b.at(i, j) || math.IsNaN(a.at(i, j)) != math.IsNaN(b.at(i, j)) {
				return false
			}
		}
	}
	return true
}

var triangularTests = []blas64.Triangular{
	{N: 3, Stride: 3, Data: []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}},
	{N: 3, Stride: 5, Data: []float64{
		1, 2, 3, 0, 0,
		4, 5, 6, 0, 0,
		7, 8, 9, 0, 0,
	}},
}

func TestConvertTriangular(t *testing.T) {
	for _, test := range triangularTests {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower, blas.All} {
			for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
				test.Uplo = uplo
				test.Diag = diag
				colmajor := NewColMajorTriangularFrom(test)
				if !equalTriangular(colMajorTriangular(colmajor), rowMajorTriangular(test)) {
					t.Errorf("unexpected result for row major to col major conversion:\n\tgot: %+v\n\tfrom:%+v",
						colmajor, test)
				}
				rowmajor := NewRowMajorTriangularFrom(colmajor)
				if !equalTriangular(rowMajorTriangular(rowmajor), rowMajorTriangular(test)) {
					t.Errorf("unexpected result for row major to col major conversion:\n\tgot: %+v\n\twant:%+v",
						rowmajor, test)
				}
			}
		}
	}
}

func TestCopyTriangularRowMajor(t *testing.T) {
	for _, test := range triangularTests {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower, blas.All} {
			for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
				src := test
				src.Uplo = uplo
				src.Diag = diag
				for stride := src.N; stride <= src.Stride+1; stride++ {
					dst := blas64.Triangular{
						N:      src.N,
						Stride: stride,
						Data:   make([]float64, src.N*stride),
						Uplo:   src.Uplo,
						Diag:   src.Diag,
					}
					for i := range dst.Data {
						dst.Data[i] = math.NaN()
					}
					CopyTriangularRowMajor(dst, src)
					if !equalTriangular(rowMajorTriangular(dst), rowMajorTriangular(src)) {
						t.Errorf("unexpected result for row major copy:\n\tgot: %+v\n\tfrom:%+v",
							dst, src)
					}
					// We are abusing the indexing to see outside the real data.
					for i := 0; i < dst.N; i++ {
						for j := dst.N; j < dst.Stride; j++ {
							if !math.IsNaN(rowMajorTriangular(dst).at(i, j)) {
								t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
									i, j, dst, src)
							}
						}
					}

					// Now check the invalid triangle for overwrites.
					for i := 0; i < dst.N; i++ {
						for j := 0; j < dst.N; j++ {
							v, ok := rowMajorTriangular(dst).atOK(i, j)
							if !ok && !math.IsNaN(v) {
								t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
									i, j, dst, src)
							}
						}
					}
				}
			}
		}
	}
}

func TestCopyTriangularColMajor(t *testing.T) {
	for _, test := range triangularTests {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower, blas.All} {
			for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
				test.Uplo = uplo
				test.Diag = diag
				src := NewColMajorTriangularFrom(test)
				for stride := src.N; stride <= src.Stride+1; stride++ {
					dst := Triangular{
						N:      src.N,
						Stride: stride,
						Data:   make([]float64, src.N*stride),
						Uplo:   src.Uplo,
						Diag:   src.Diag,
					}
					for i := range dst.Data {
						dst.Data[i] = math.NaN()
					}
					CopyTriangularColMajor(dst, src)
					if !equalTriangular(colMajorTriangular(dst), colMajorTriangular(src)) {
						t.Errorf("unexpected result for col major copy:\n\tgot: %+v\n\tfrom:%+v",
							dst, src)
					}
					// We are abusing the indexing to see outside the real data.
					for i := dst.N; i < dst.Stride; i++ {
						for j := 0; j < dst.N; j++ {
							if !math.IsNaN(colMajorTriangular(dst).at(i, j)) {
								t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
									i, j, dst, src)
							}
						}
					}

					// Now check the invalid triangle for overwrites.
					for i := 0; i < dst.N; i++ {
						for j := 0; j < dst.N; j++ {
							v, ok := colMajorTriangular(dst).atOK(i, j)
							if !ok && !math.IsNaN(v) {
								t.Errorf("unexpected result for row major copy value overwritten at (%d,%d):\n\tgot: %+v\n\tfrom:%+v",
									i, j, dst, src)
							}
						}
					}
				}
			}
		}
	}
}
