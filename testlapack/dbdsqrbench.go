// Copyright ©2015 The gonum Authors. All rights reserved.
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
	Dgebrder
}

func DbdsqrBench(b *testing.B, impl DbdsqrerDgebrder, m, n, lda int, useVT bool) {
	// Construct a random bidiagonal matrix by applying Dgebrd on a random matrix.
	// The benchmarking section of netlib, found at http://www.netlib.org/lapack/lug/node71.html
	// indicates that benchmarks are typically done on random matricies that are
	// transformed to produce desired properties (symmetry, diagonalization, etc.).

	if lda == 0 {
		lda = n
	}

	uplo := blas.Upper
	if m < n {
		uplo = blas.Lower
	}

	minmn := min(m, n)
	a := make([]float64, m*lda)
	d := make([]float64, minmn)
	e := make([]float64, minmn-1)
	tauP := make([]float64, minmn)
	tauQ := make([]float64, minmn)

	work := make([]float64, 1)

	impl.Dgebrd(m, n, a, lda, d, e, tauP, tauQ, work, -1)
	work = make([]float64, int(work[0]))
	lwork := len(work)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()

		// TODO(jonlawlor): When useVT is false, this initialization is a large
		// portion of the benchmark time.  Because the number of iterations in a
		// benchmark is inversely proportional to the time of a single iteration,
		// this causes a large delay, because Dbdsqr is so fast, and the initialization
		// time is ignored.  Maybe there is a way to amortize the cost of constructing
		//  a random bidirectional matrix?
		for i := range a {
			a[i] = rand.NormFloat64()
		}

		impl.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, lwork)
		if useVT {
			// Path might be used by dgesvd.
			pt := make([]float64, n*n)
			ldpt := n
			for i := 0; i < n; i++ {
				pt[i*ldpt+i] = 1
			}

			b.StartTimer()
			impl.Dbdsqr(uplo, minmn, minmn, 0, 0, d, e, pt, minmn, nil, 0, nil, 0, work)
			continue
		}
		b.StartTimer()
		impl.Dbdsqr(uplo, minmn, 0, 0, 0, d, e, nil, 0, nil, 0, nil, 0, work)

	}
}
