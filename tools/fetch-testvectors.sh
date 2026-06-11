#!/bin/sh
# Fetches the official Opus decoder conformance vectors (RFC 8251 update)
# into tests/vectors/. ~121 MB extracted; not committed to the repository.
set -eu
cd "$(dirname "$0")/.."
mkdir -p tests/vectors
curl -sL "https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz" \
    | tar xzf - --strip-components=1 -C tests/vectors
echo "Test vectors ready in tests/vectors/"
