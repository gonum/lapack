#!/bin/bash
set -ex

cd $GOPATH/src/github.com/$TRAVIS_REPO_SLUG/cgo/clapack
go install -v -x
