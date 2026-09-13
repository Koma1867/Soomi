# Soomi Chess Engine (Go)

<p align="center">
  <img src="Soomi.png" alt="Soomi Chess Engine Logo" width="1000">
</p>

<h1 align="center">Soomi Chess Engine</h1>

Soomi is a UCI chess engine written in Go, in a single file.

## Features
- Magic bitboards
- Negamax with alpha-beta, PVS, iterative deepening, aspiration windows
- Transposition table, check extensions, IIR, mate distance pruning
- Quiescence search with delta pruning
- SEE
- Pruning: LMR, LMP, NMP, RFP, ProbCut, SEE pruning
- Move ordering: hash move, MVV-LVA, SEE, killers, history, countermoves
- Tapered evaluation: material, PST, mobility, king safety, king tropism, pawn structure, pawn storms, outposts, bishop pair, rooks on open files, tempo
- Pawn hash table
- Texel-tuned evaluation
- Dual-bound time management
- UCI info: seldepth, hashfull, currmove, lowerbound/upperbound

## Limitations
- Single-threaded, no pondering, no opening book

## Build
Go 1.21+: https://go.dev/doc/install

```bash
go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi.exe Soomi.go
```

## Possible improvements
- SPSA tuning of search parameters
- Multithreading
- Pondering, opening book
- More evaluation terms
- Better move ordering

## License
Free to distribute and modify. Please credit the original author (Otto Laukkanen) if you use this code.

## Acknowledgments
- Maksim Korzh for the mention on his BBC (BitBoard Chess) engine's GitHub page.
- Chess Programming Wiki for the good information.
