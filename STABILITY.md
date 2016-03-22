# API Stability Policy

We will not change the gonum/lapack/... packages' exported API in backward incompatible ways.
Future changes to these package will not break dependent code.

## Scope

This document is similar to and inspired by the [Go 1 compatibility
promise](https://golang.org/doc/go1compat) via Go kit [RFC007](https://github.com/go-kit/kit/blob/master/rfc/rfc007-api-stability.md).

### Coverage

The promise of stability includes:

* The package name,
* Exported type declarations and struct fields (names and types),
* Exported function and method names, parameters, and return values,
* Exported constant names and values,
* Exported variable names and values,
* The documented behavior of all exported code.

### Exceptions

* Security. A security issue in the package may come to light whose resolution
  requires breaking compatibility. We reserve the right to address such security
  issues.

* Unspecified behavior. Programs that depend on unspecified behavior may break
  in future releases.

* Bugs. If the package has a bug, a program that depends on the buggy behavior
  may break if the bug is fixed. We reserve the right to fix such bugs.

* Method sets in github.com/gonum/lapack. The method sets of lapack.Float64 and
  lapack.Complex128 may be extended.

* Dot import of the lapack/native package. If a program imports gonum/lapack/native
  using import . "github.com/gonum/lapack/native", additional names later defined
  in gonum/lapack/native may conflict with other names defined in the program.

* Methods marked internal in "github.com/gonum/lapack/native" are not covered by this
  API stability promise. Methods that are not excluded are defined in the lapack.Float64
  and lapack.Complex128 interface method sets.

* The package "github.com/gonum/lapack/cgo/clapack". This package has the same
  promise of stability as the "syscall" package in the standard library.