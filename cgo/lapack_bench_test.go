// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgo

import (
	"testing"

	"github.com/gonum/lapack/testlapack"
)

func BenchmarkDbdsqr10x10(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 10, 10, 0, false)
}

func BenchmarkDbdsqr50x50(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 50, 50, 0, false)
}

func BenchmarkDbdsqr100x100(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 100, 100, 0, false)
}

func BenchmarkDbdsqr200x200(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 200, 200, 0, false)
}

func BenchmarkDbdsqr1000x1000(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 1000, 1000, 0, false)
}

func BenchmarkDbdsqrWithVT10x10(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 10, 10, 0, true)
}

func BenchmarkDbdsqrWithVT50x50(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 50, 50, 0, true)
}

func BenchmarkDbdsqrWithVT100x100(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 100, 100, 0, true)
}

func BenchmarkDbdsqrWithVT200x200(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 200, 200, 0, true)
}

func BenchmarkDbdsqrWithVT1000x1000(b *testing.B) {
	testlapack.DbdsqrBench(b, impl, 1000, 1000, 0, true)
}
