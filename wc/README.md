# wc

This example implements a word/line/byte counting command similar to the POSIX `wc` command.

This example use a class, a struct, and Foundation file I/O APIs. It also uses `guard`, which is a nice way to bind a variable to a scope where `if` wouldn't be able to do the same easily.

# Example

```
$ swift run wc README.md 
Building for debugging...
[1/1] Write swift-version--1D488833B531E0A.txt
Build complete! (0.09s)
counts: Counts(bytes: 481, words: 78, lines: 11)
```
