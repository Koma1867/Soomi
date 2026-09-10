# Soomi Chess Engine (Go)

<p align="center">
  <img src="Soomi_logo.png" alt="Soomi Chess Engine Logo" width="1000">
</p>

<h1 align="center">Soomi Chess Engine</h1>

-Soomi is a small chess engine written in Go. It started as a learning project to understand core chess programming techniques like bitboards and search algorithms.

## Features
- Bitboard move generation using Magic Bitboards
- Evaluation based on material, PST, mobility, king safety, pawn structure, tempo, king tropism, outposts, pawn storms, bishop pair.
- Tapered evaluation
- Negamax search with Alpha-Beta pruning
- Principal Variation Search (PVS)
- Iterative Deepening with Aspiration Windows
- Transposition Table
- Check extensions
- Quiescence Search with Delta Pruning
- Static Exchange Evaluation (SEE)
- Pruning techniques including Late Move Reductions (LMR), Null Move Pruning (NMP), Reverse Futility Pruning (RFP), Late Move Pruning (LMP) and Mate Distance Pruning
- Move ordering using Hash move, MVV-LVA and SEE (captures), killers, history heuristic and countermove heuristic
- UCI protocol (compatible with Arena, others not tested)
- FEN parsing
- Simple time management with score loss, bestmove, soft / hard bounds
- Texel tuned eval on own generated data

## Limitations
- Single-threaded: The engine runs on a single core and does not support pondering.

## Build & Run
Install Go (1.20+ recommended): https://go.dev/doc/install

To build, run one of the following in your terminal:

```bash
# Standard build
go build -o Soomi.exe soomi.go

# Optimized/Smaller build (strips debug info)
go build -trimpath -ldflags "-s -w" -o Soomi.exe soomi.go

# Disabling go bounds checking, perhaps 5% faster
go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi.exe soomi.go
```

## Possible improvements
- Tune pruning values
- Add pondering, opening book
- Multithreading
- More evaluation terms
- Better move-ordering

## License
Free to distribute and modify. Please credit the original author (Otto Laukkanen) if you use this code.

## Acknowledgments
- Maksim Korzh for the mention on his BBC (BitBoard Chess) engine's GitHub page.
- Chess Programming Wiki for the good information.
