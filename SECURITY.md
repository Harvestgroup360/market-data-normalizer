# Security policy

## Scope

`market-data-normalizer` is a pure-Python library with no runtime
dependencies. It parses files and byte streams you hand it, and it makes no
network calls of its own. That shape means the realistic vulnerability classes
are:

- a malformed input that causes unbounded memory or CPU use rather than an
  exception,
- a parsing path that raises something other than a documented error type,
- anything that lets input data reach `eval`, the filesystem or a subprocess.

The last one should be impossible: the package imports nothing that could do
it. If you find otherwise, that is exactly what we want to hear about.

## Reporting

Please report privately first, to **github@harvestgroup360.com**, rather than
opening a public issue. Include the input that triggers it and the version.

We will confirm receipt within three working days and tell you what we intend
to do. If we disagree that it is a security issue we will say so and why,
rather than letting it go quiet.

## Supported versions

The latest release on PyPI is supported. Given the size of the package and the
release cadence, back-porting to older versions is not something we offer —
upgrading is a single version bump with a changelog entry for every step.

## What is not a vulnerability

Data that is wrong because the file was wrong. The library reports what it
found, including where it refused to answer, and a great deal of it exists to
make bad input visible rather than to make it disappear.
