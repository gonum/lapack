// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgo

import (
	"testing"

	"github.com/gonum/lapack/testlapack"
)

func BenchmarkDbdsqr100x100(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 100, 100, 0, false)
}

func BenchmarkDbdsqr200x200(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 200, 200, 0, false)
}

func BenchmarkDbdsqr300x300(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 300, 300, 0, false)
}

func BenchmarkDbdsqrWithVT100x100(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 100, 100, 0, true)
}

func BenchmarkDbdsqrWithVT200x200(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 200, 200, 0, true)
}

func BenchmarkDbdsqrWithVT300x300(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 300, 300, 0, true)
}
