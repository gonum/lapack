#!/bin/bash
set -ex

go env
go get -d -t -v ./...
go test -v ./...
if [[ $TRAVIS_SECURE_ENV_VARS = "true" ]]; then bash -c "$GOPATH/src/github.com/$TRAVIS_REPO_SLUG/.travis/test-coverage.sh"; fi
