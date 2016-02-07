// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math/rand"
	"testing"

	"github.com/gonum/blas"
)

type DbdsqrerDgebrder interface {
	Dbdsqrer
	Dgebrd(m, n int, a []float64, lda int, d, e, tauQ, tauP, work []float64, lwork int)
}

func DbdsqrBench(b *testing.B, impl DbdsqrerDgebrder, m, n, lda int, useVT bool) {
	// Construct a random bidiagonal matrix by applying Dgebrd on a random matrix.
	// The benchmarking section of netlib, found at http://www.netlib.org/lapack/lug/node71.html
	// indicates that benchmarks are typically done on random matricies that are
	// transformed to produce desired properties (symmetry, diagonalization, etc.).
	rand.Seed(1)

	if lda == 0 {
		lda = n
	}

	uplo := blas.Upper
	if m < n {
		uplo = blas.Lower
	}

	minmn := min(m, n)
	a := make([]float64, m*lda)
	for i := range a {
		a[i] = rand.NormFloat64()
	}
	d := make([]float64, minmn)
	e := make([]float64, minmn-1)
	tauP := make([]float64, minmn)
	tauQ := make([]float64, minmn)

	work := make([]float64, 1)

	impl.Dgebrd(m, n, a, lda, d, e, tauP, tauQ, work, -1)

	work = make([]float64, int(work[0]))
	lwork := len(work)

	// Construct the bidiagonal matrix in d and e.
	impl.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, lwork)
	dCopy := make([]float64, len(d))
	copy(d, dCopy)
	eCopy := make([]float64, len(e))
	copy(e, dCopy)

	var pt []float64
	var ptCopy []float64

	if useVT {
		// Path might be used by dgesvd.
		pt = make([]float64, n*n)
		for i := 0; i < n; i++ {
			pt[i*n+i] = 1
		}
		ptCopy = make([]float64, len(pt))
		copy(pt, ptCopy)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		copy(dCopy, d)
		copy(eCopy, e)

		if useVT {
			// Path might be used by dgesvd.
			copy(ptCopy, pt)
			b.StartTimer()
			impl.Dbdsqr(uplo, minmn, minmn, 0, 0, d, e, pt, minmn, nil, 0, nil, 0, work)
			continue
		}
		b.StartTimer()
		impl.Dbdsqr(uplo, minmn, 0, 0, 0, d, e, nil, 0, nil, 0, nil, 0, work)
	}
}
