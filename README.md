# Soomi Chess Engine (Go)

<p align="center">
  <img src="Soomi.png" alt="Soomi Chess Engine Logo" width="1000">
</p>

<h1 align="center">Soomi Chess Engine</h1>

Soomi is a UCI chess engine written in Go, in a single file plus its neural network (`soomi.nnue`), which is embedded into the executable. From version 1.2.0B forward written with Claude code.

## Features
- Magic bitboards
- Negamax with alpha-beta, PVS, iterative deepening, aspiration windows
- Transposition table, check extensions, IIR, mate distance pruning
- Quiescence search with delta pruning
- SEE
- Pruning: LMR, LMP, NMP, RFP, ProbCut, SEE pruning
- Move ordering: hash move, MVV-LVA, SEE, killers, history, countermoves
- NNUE evaluation: 768 -> 128x2 -> 8 output buckets, horizontal king mirroring, trained on Soomi's own self-play games
- AVX2 SIMD inference and lazy accumulator updates
- Dual-bound time management
- UCI info: seldepth, hashfull, currmove, lowerbound/upperbound

## Limitations
- Single-threaded, no pondering, no opening book
- x86-64 only: Go's SIMD package supports only AMD64. CPUs without AVX2 run a slower fallback.

## Build
Go 1.27+: https://go.dev/doc/install

`soomi.nnue` must be in the same folder as `Soomi.go`. Soomi uses Go's experimental SIMD package, so set `GOEXPERIMENT=simd` before building.

Windows (Command Prompt):
```bat
set GOEXPERIMENT=simd
go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi.exe Soomi.go
```

Linux:
```bash
GOEXPERIMENT=simd go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o soomi Soomi.go
```

## Tests
`commands\run_tests.bat` runs perft, the fixed-depth bench signature, FEN validation, long games and search edge cases. Elsewhere, with `GOEXPERIMENT=simd` set:
```bash
go test Soomi.go soomi_test.go
```

## IMPORTANT
From version 1.2.0B forwards built with Claude Code, do not test further, focus on human engines! Im updating github so i can have a backup for the files.

## Possible improvements
- SPSA tuning of search parameters
- Multithreading
- Pondering, opening book
- Larger networks, more training data
- Better move ordering

## Changes
See `changelog.txt`.

## License
Free to distribute and modify. Please credit the original author (Otto Laukkanen) if you use this code.

## Acknowledgments
- Maksim Korzh for the mention on his BBC (BitBoard Chess) engine's GitHub page.
- Chess Programming Wiki for the good information.
