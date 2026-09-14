package main

import (
	"bufio"
	_ "embed"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/bits"
	"os"
	"simd/archsimd"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	White  = 0
	Black  = 1
	Pawn   = 0
	Knight = 1
	Bishop = 2
	Rook   = 3
	Queen  = 4
	King   = 5
)

/*
  ----------------------------------------------------------------------------------
   BITBOARD REPRESENTATION
  ----------------------------------------------------------------------------------
   The board is represented as a series of 64-bit integers, where each bit corresponds to a square on the chess board.

   Visual Mapping

   Rank 8 | 56 57 58 59 60 61 62 63
   Rank 7 | 48 49 50 51 52 53 54 55
   Rank 6 | 40 41 42 43 44 45 46 47
   Rank 5 | 32 33 34 35 36 37 38 39
   Rank 4 | 24 25 26 27 28 29 30 31
   Rank 3 | 16 17 18 19 20 21 22 23
   Rank 2 |  8  9 10 11 12 13 14 15
   Rank 1 |  0  1  2  3  4  5  6  7
           -----------------------
             A  B  C  D  E  F  G  H
*/

const (
	EngineName          = "Soomi V1.3.0"
	MaxDepth            = 50
	Infinity            = 30000
	Mate                = 29000
	MateScoreGuard      = 1000
	MaxGamePly          = 1024 // history entries: game plies (see setPosition) plus MaxDepth search plies
	NodeCheckMaskSearch = 1023
	DefaultTTSizeMB     = 256
	ZobristSeed         = 1070372
	TotalPhase          = 24
	EndgamePhase        = 18 // isEndgame: more phase than this is gone
)

// Search parameters: every depth limit, margin and reduction the search uses.
const (
	AspirationStartDepth = 4
	AspirationBase       = 50   // first window half-width, doubled after each fail
	AspirationMaxWindow  = 1000 // once the window reaches this, search the full window
	IIRDepthMin          = 4    // reduce by 1 ply without a hash move at depth >= this
	RFPDepthMax          = 8
	RFPMargin            = 150 // per ply of depth
	NMPDepthMin          = 3
	NMPReductionBase     = 3
	NMPReductionDiv      = 6 // R = NMPReductionBase + depth/NMPReductionDiv
	ProbCutDepthMin      = 5
	ProbCutMargin        = 200
	ProbCutReduction     = 4
	LMRMinChildDepth     = 3
	LMRLateMoveAfter     = 3
	LMRBase              = 0.5
	LMRDivisor           = 3.0 // reduction = LMRBase + ln(depth) * ln(move number) / LMRDivisor
	LMPDepthMax          = 4   // LMP only at depth <= this
	LMPBase              = 3   // quiet limit: LMPBase + depth * depth
	SEEPruneDepthMax     = 8   // SEE prune only at depth <= this
	SEEQuietCoeff        = 80  // quiet margin: -coeff * depth
	SEENoisyCoeff        = 30  // capture margin: -coeff * depth * depth
	DeltaMargin          = 150 // quiescence delta pruning
	MaxHistory           = 16384
	HistoryBonusMax      = 400
)

// Move ordering scores.
const (
	ScoreHash        = 1000000
	ScorePromoBase   = 900000
	ScoreCaptureBase = 800000
	ScoreKiller1     = 750000
	ScoreKiller2     = 740000
	ScoreCountermove = 200000
	MVVLVAWeight     = 100
)

// Time management (see allocateTime).
const (
	DefaultMovesToGo                    = 20
	DefaultMoveOverheadMs int64         = 20 // UCI Move Overhead: clock kept back for GUI and pipe lag
	MinTimeMs             int64         = 5  // shortest search time ever planned
	ContinueMargin        time.Duration = 10 * time.Millisecond
	TMIncrementPct                      = 75  // the soft limit adds this much of the increment
	TMHardLimitPct                      = 350 // the hard limit is this much of the soft limit
	TMScaleMinDepth                     = 5   // stability and score-drop scaling start at this depth
	TMStableIterations                  = 3
	TMStableScale                       = 0.75 // best move unchanged for TMStableIterations
	TMUnstableScale                     = 1.25 // best move changed this iteration
	TMScoreDrop                         = 40   // a score this far below the previous iteration...
	TMScoreDropScale                    = 1.20 // ...scales the soft limit by this
	TMNextIterationFactor               = 2    // the next iteration is expected to take this many times the last
	TMSoftOverrunPct                    = 150  // do not start an iteration expected to end past this much of the soft limit
)

const (
	FlagQuiet         = 0
	FlagCapture       = 4
	FlagEP            = 5
	FlagCastle        = 2
	FlagPromoN        = 8 // promotion flags: 8-11 quiet N,B,R,Q; 12-15 capturing N,B,R,Q
	FlagPromoQ        = 11
	FlagPromoCN       = 12
	FlagPromoCQ       = 15
	TTFlagExact uint8 = 0
	TTFlagLower uint8 = 1
	TTFlagUpper uint8 = 2
)

var (
	pieceValues       = [6]int{89, 313, 317, 504, 1001, 20000}
	pst               [2][6][64]int
	piecePhase        = [6]int{0, 1, 1, 2, 4, 0}
	tt                *TranspositionTable
	rookMagics        [64]MagicEntry
	bishopMagics      [64]MagicEntry
	rookAttackTable   [102400]Bitboard
	bishopAttackTable [5248]Bitboard
	lmrTable          [MaxDepth + 1][256]int
	lvaOrder          = [6]int{Pawn, Bishop, Knight, Rook, Queen, King}
	castleMask        [64]int
)

/*
  ----------------------------------------------------------------------------------
   MAGIC BITBOARDS
  ----------------------------------------------------------------------------------
   Magic bitboards give sliding piece attacks with a single table lookup

   The concept goes as follows
   1. Mask     A. Isolate relevant blocker squares for a piece at square X.
   2. Multiply B. Occupancy * MagicNumber (this scatters bits into high-order positions).
   3. Shift    C. Right shift to extract an index.
   4. Lookup   D. Use index to fetch pre-calculated attack set from a table.

   [ Occupancy on Board ]  &  [ Movement Mask ]
             |
             v
      [ Masked Blockers ]  * [ Magic Number ]
             |
             v
      [   Scattered Bits (Hash)   ]  >>  (64 - BitsInMask)
             |
             v
        [ Index ]  --->  [ Attack Table ]  --->  [ Attack Bitboard ]
*/

type MagicEntry struct {
	mask   Bitboard
	magic  Bitboard
	shift  uint8
	offset uint32
}

var rookMagicNumbers = [64]uint64{
	0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
	0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
	0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
	0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
	0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
	0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
	0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
	0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
	0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
	0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
	0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
	0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
	0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
	0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
	0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
	0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2,
}

var bishopMagicNumbers = [64]uint64{
	0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
	0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
	0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
	0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
	0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
	0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
	0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
	0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
	0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
	0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
	0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
	0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
	0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
	0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
	0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
	0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200,
}

/*
  ----------------------------------------------------------------------------------
   MOVE BIT-PACKING
  ----------------------------------------------------------------------------------
   To save memory and increase speed, a Move is not a struct, but a single 32-bit integer.
   We pack the 'From' square, 'To' square, and 'Flags' (promotion/capture type) into specific bits.

   [ 31 ... 16 ] [ 15 14 13 12 ] [ 11 10 9 8 7 6 ] [ 5 4 3 2 1 0 ]
     (Unused)       Flag (4b)       To Sq (6b)      From Sq (6b)
                       ^                 ^                ^
                       |                 |                |
   Example:           0100 (Capture)    111000 (H8)      000000 (A1)

   - Mask 0x3F (63) extracts squares (0-63).
   - Shift >> 6 moves the 'To' bits to the bottom.
   - Shift >> 12 moves the 'Flag' bits to the bottom.
*/

type Bitboard uint64
type Move uint32

type Position struct {
	pieces           [2][6]Bitboard
	occupied         [2]Bitboard
	all              Bitboard
	side             int
	castle           int
	epSquare         int
	halfmove         int
	hash             uint64
	square           [64]int
	historyKeys      [MaxGamePly]uint64
	acc              [MaxGamePly][2][NNHidden]int16 // NNUE accumulators, indexed like historyKeys
	nn               [MaxGamePly]NNPly              // each ply's accumulator change, applied lazily
	historyPly       int
	lastIrreversible int
	kingSq           [2]int
	phase            int
}

type Undo struct {
	castle           int
	epSquare         int
	halfmove         int
	captured         int
	lastIrreversible int
}

type TimeControl struct {
	wtime     int64
	btime     int64
	winc      int64
	binc      int64
	movestogo int
	movetime  int64
	infinite  bool
	depth     int
	optimumMs int64
	deadline  time.Time
	stopped   int32
}

type SearchStack struct {
	killer1 Move
	killer2 Move
	pv      [MaxDepth]Move // principal variation from this ply (PV nodes only)
	pvLen   int
}

// Searcher is one search thread: its own copy of the position, the move ordering
// tables it learns over a game, and the counters of the current search. The
// transposition table is shared.
type Searcher struct {
	pos          Position
	out          *uciWriter // info lines go here
	tc           *TimeControl
	ss           [MaxDepth + 1]SearchStack
	history      [2][64][64]int
	countermoves [2][64][64]Move
	nodes        int64
	seldepth     int
	start        time.Time
}

// phase counts the non-pawn material that is gone (N=1, B=1, R=2, Q=4): 0 with
// every piece on the board, TotalPhase (24) with none. Search uses it to spot endgames.
func (p *Position) computePhase() {
	p.phase = TotalPhase
	for pt := Knight; pt <= Queen; pt++ {
		p.phase -= bits.OnesCount64(uint64(p.pieces[White][pt]|p.pieces[Black][pt])) * piecePhase[pt]
	}
}

func (p *Position) isEndgame() bool {
	return p.phase > EndgamePhase
}

var (
	rookDirs   = [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	bishopDirs = [4][2]int{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
)

// rayAttacks walks each direction from sq to the board edge or the first
// blocker in occ; the blocker's square is included.
func rayAttacks(sq int, occ Bitboard, dirs *[4][2]int) Bitboard {
	var attacks Bitboard
	for _, d := range dirs {
		for r, f := sq/8+d[0], sq%8+d[1]; r >= 0 && r < 8 && f >= 0 && f < 8; r, f = r+d[0], f+d[1] {
			bb := Bitboard(1) << (r*8 + f)
			attacks |= bb
			if occ&bb != 0 {
				break
			}
		}
	}
	return attacks
}

// relevantMask is the magic occupancy mask: every ray square except the last
// one before the edge, since a piece on the edge square blocks nothing.
func relevantMask(sq int, dirs *[4][2]int) Bitboard {
	var mask Bitboard
	for _, d := range dirs {
		for r, f := sq/8+d[0], sq%8+d[1]; r+d[0] >= 0 && r+d[0] < 8 && f+d[1] >= 0 && f+d[1] < 8; r, f = r+d[0], f+d[1] {
			mask |= Bitboard(1) << (r*8 + f)
		}
	}
	return mask
}

func initMagicBitboards() {
	initMagics(&rookMagics, rookAttackTable[:], &rookMagicNumbers, &rookDirs)
	initMagics(&bishopMagics, bishopAttackTable[:], &bishopMagicNumbers, &bishopDirs)
}

// initMagics fills one slider's entries and attack table, enumerating every
// subset of each mask with the Carry-Rippler trick.
func initMagics(entries *[64]MagicEntry, table []Bitboard, numbers *[64]uint64, dirs *[4][2]int) {
	offset := uint32(0)
	for sq := 0; sq < 64; sq++ {
		mask := relevantMask(sq, dirs)
		bitCount := bits.OnesCount64(uint64(mask))
		e := &entries[sq]
		*e = MagicEntry{mask: mask, magic: Bitboard(numbers[sq]), shift: 64 - uint8(bitCount), offset: offset}
		for occ := Bitboard(0); ; {
			table[e.offset+uint32((occ*e.magic)>>e.shift)] = rayAttacks(sq, occ, dirs)
			if occ = (occ - mask) & mask; occ == 0 {
				break
			}
		}
		offset += 1 << bitCount
	}
}

var sqBB [64]Bitboard

func initSqBB() {
	for i := 0; i < 64; i++ {
		sqBB[i] = Bitboard(1) << uint(i)
	}
}

var (
	zobristPiece      [2][6][64]uint64
	zobristSide       uint64
	zobristCastleDiff [16]uint64
	zobristEP         [8]uint64
	knightAttacks     [64]Bitboard
	kingAttacks       [64]Bitboard
	pawnAttacks       [2][64]Bitboard
)

func (p *Position) isRepetition() bool {
	target := p.hash
	for i := p.historyPly - 2; i >= p.lastIrreversible; i -= 2 {
		if p.historyKeys[i] == target {
			return true
		}
	}
	return false
}

func (p *Position) isInsufficientMaterial() bool {
	return (p.pieces[White][Pawn]|p.pieces[Black][Pawn]|p.pieces[White][Rook]|p.pieces[Black][Rook]|p.pieces[White][Queen]|p.pieces[Black][Queen]) == 0 && bits.OnesCount64(uint64(p.occupied[White])) <= 2 && bits.OnesCount64(uint64(p.occupied[Black])) <= 2
}

type TTEntry struct {
	key    uint64
	packed uint64
}

/*
  ----------------------------------------------------------------------------------
   TRANSPOSITION TABLE (TT)
  ----------------------------------------------------------------------------------
   A hash map that stores search results. It uses Zobrist Hashing, where the
   board state is XORed with random 64-bit numbers.

   Structure of an Entry (Packed into 64 bits):
   [    Move (32b)    ] [ Score (16b) ] [ Gen (8b) ] [ Depth (6b) ] [ Flag (2b) ]
   ^
   |
   (Upper 32 bits store the Move)

   Lookup Process:
   1. Compute Zobrist Hash of current position.
   2. Index = Hash & (TableSize - 1); the size is a power of 2.
   3. Check if stored Key matches current Key.
   4. If Depth >= NeededDepth, use the stored Score immediately.
*/

type TranspositionTable struct {
	entries []TTEntry
	mask    uint64
	gen     uint32
}

// initTT initializes the transposition table.
// Small note: it is recommended to only use power of 2 values
// As the engine will automatically round it down to the nearest power of 2 anyways.
// Example: You input setoption hash value 150. It will be rounded to 128.
// Recommended values: 64, 128, 256, 512, 1024...
func initTT(sizeMB int) {
	// One TTEntry is two uint64s = 16 bytes. bits.Len64 finds how many bits the entry
	// count needs; shifting 1 by one less rounds it down to a power of 2 (100 -> 64).
	// size-1 is then a mask, so indexing can use AND instead of the much slower modulo.
	size := uint64(1) << (bits.Len64(uint64(sizeMB)*1024*1024/16) - 1)
	tt = &TranspositionTable{entries: make([]TTEntry, size), mask: size - 1}
}

func (t *TranspositionTable) newGeneration() {
	t.gen++
	if t.gen > 255 {
		t.gen = 1
		for i := range t.entries {
			t.entries[i] = TTEntry{}
		}
	}
}

// probe decodes the packed word in place (layout above), which keeps it inlinable.
func (t *TranspositionTable) probe(key uint64, minDepth int) (move Move, score int, flag uint8, found, usable bool) {
	e := t.entries[key&t.mask]
	if e.key != key {
		return
	}
	p := e.packed
	return Move(p >> 32), int(int16(p >> 16)), uint8(p & 3), true, uint8(p>>8) == uint8(t.gen) && int(p>>2&0x3F) >= minDepth
}

// save stores a search result, always replacing the old entry: simple, but a deep
// result can be overwritten by a shallow one.
func (t *TranspositionTable) save(key uint64, mv Move, score int, depth int, flag uint8) {
	packed := uint64(mv)<<32 | uint64(uint16(int16(score)))<<16 | uint64(uint8(t.gen))<<8 | uint64(uint8(min(depth, 63)))<<2 | uint64(flag&3)
	t.entries[key&t.mask] = TTEntry{key: key, packed: packed}
}

// scoreToTT converts a mate score from distance-to-root to distance-to-this-node,
// so a stored mate stays correct when the position is reached at another ply.
func scoreToTT(score, ply int) int {
	if score > Mate-MateScoreGuard {
		return score + ply
	}
	if score < -Mate+MateScoreGuard {
		return score - ply
	}
	return score
}

// hashfull samples the first 1000 entries (the table always has at least 65536).
func (t *TranspositionTable) hashfull() int {
	used := 0
	for _, e := range t.entries[:1000] {
		if e.key != 0 && uint8(e.packed>>8) == uint8(t.gen) {
			used++
		}
	}
	return used
}

func init() {
	initCastleMask()
	initPST()
	initZobrist()
	initSqBB()
	initAttacks()
	initMagicBitboards()
	initLMR()
	initTT(DefaultTTSizeMB)
}

func initCastleMask() {
	for i := 0; i < 64; i++ {
		castleMask[i] = 15
	}
	castleMask[0], castleMask[7] = 13, 14
	castleMask[56], castleMask[63] = 7, 11
	castleMask[4], castleMask[60] = 12, 3
}

func initLMR() {
	for d := 1; d <= MaxDepth; d++ {
		for m := 1; m < 256; m++ {
			val := LMRBase + math.Log(float64(d))*math.Log(float64(m))/LMRDivisor
			lmrTable[d][m] = int(val)
		}
	}
}

// pst is only used to order quiet moves that have no history yet (orderMoves).
func initPST() {
	pst[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		-14, 12, -3, -13, -9, 9, 16, -8,
		2, 4, 2, -3, 3, 1, 10, 3,
		-8, -7, 10, 22, 24, 19, 2, -5,
		2, 8, 20, 30, 29, 31, 23, 4,
		6, 6, 10, 39, 43, 6, 21, 10,
		63, 28, 94, 173, 126, 8, 25, 14,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pst[White][Knight] = [64]int{
		-11, 3, 1, 9, 12, 0, 3, -81,
		7, 0, 8, 20, 24, 14, 7, 9,
		-9, 13, 15, 24, 29, 19, 27, -1,
		2, 29, 24, 24, 18, 22, 7, 19,
		29, 12, 25, 24, 28, 44, 19, 28,
		20, 44, 33, 61, 56, 49, 32, 47,
		-20, -12, 15, 68, 39, 46, -9, 52,
		-165, -89, -9, -15, 40, -12, -36, -77,
	}

	pst[White][Bishop] = [64]int{
		-15, 12, 5, -18, 3, -6, -19, -9,
		14, 28, 6, 10, 9, 12, 25, -22,
		15, 14, 17, 13, 6, 14, 7, -2,
		0, 3, 3, 27, 33, 0, -15, 11,
		-7, 4, 26, 35, 24, 29, 6, -27,
		-7, 14, 7, 37, 20, 51, 14, -16,
		-14, -5, -10, -4, 15, -18, -44, -12,
		-34, -22, -76, -20, -76, -39, -9, -12,
	}

	pst[White][Rook] = [64]int{
		4, 7, 15, 12, 17, 11, -6, 0,
		-19, -17, -28, -16, -8, -2, -2, -29,
		-24, -6, -27, -17, -15, -18, -12, -19,
		-29, -18, -32, -5, 0, -13, -11, -17,
		-11, -1, 23, 2, 38, 16, 8, -8,
		-1, 50, 28, 31, 51, 34, 42, 6,
		12, 0, 19, 18, 40, 32, 10, 28,
		24, 42, 52, 11, 47, 51, 59, 45,
	}

	pst[White][Queen] = [64]int{
		-10, 9, 9, 5, 15, -10, -5, -11,
		2, 14, 7, 8, 7, 5, 25, 11,
		-8, 3, -6, -5, -8, -4, -2, 6,
		-5, -13, -18, -20, -13, -8, -24, -7,
		-8, -21, -12, -18, -1, -12, -11, 0,
		-19, -6, -13, 5, -2, 47, 7, 1,
		-46, -49, -4, -10, 3, 46, 0, 26,
		-26, 29, 57, 32, 41, 6, 44, 26,
	}

	pst[White][King] = [64]int{
		18, 21, 18, -28, -5, -14, 5, 3,
		43, 15, -7, -15, -12, -7, 12, -3,
		-36, 10, -9, -6, -25, -15, -25, -26,
		-7, 22, 12, -31, -14, 7, -17, -30,
		13, 43, 20, -88, -39, 20, 29, -20,
		-19, 61, 59, 33, -15, 47, -4, 6,
		0, 132, 119, 25, 62, 104, 84, 52,
		-102, 144, 79, 112, 140, 220, 130, -34,
	}

	// Mirror for black
	for pt := 0; pt < 6; pt++ {
		for sq := 0; sq < 64; sq++ {
			pst[Black][pt][sq] = pst[White][pt][sq^56]
		}
	}
}

/*
  ----------------------------------------------------------------------------------
   ZOBRIST HASHING INITIALIZATION
  ----------------------------------------------------------------------------------
   We assign a random 64-bit number to every possible board state component.

   Components Hashed:
   1. Piece at Square (e.g., White Pawn on E4)
   2. Side to Move (White/Black)
   3. Castling Rights (KQkq)
   4. En Passant File

   The final Board Hash is the XOR sum of all active components.
   Hash = [WP_on_E4] ^ [BK_on_E8] ^ [WhiteToMove] ^ ...

   Incremental Update:
   When a piece moves, we don't recalculate from scratch. We XOR out the old
   piece and XOR in the new one.
*/

func initZobrist() {
	rng := uint64(ZobristSeed)
	next := func() uint64 {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		return rng
	}

	for c := 0; c < 2; c++ {
		for pt := 0; pt < 6; pt++ {
			for sq := 0; sq < 64; sq++ {
				zobristPiece[c][pt][sq] = next()
			}
		}
	}
	zobristSide = next()
	castleKeys := [4]uint64{next(), next(), next(), next()} // K, Q, k, q
	for i := 0; i < 8; i++ {
		zobristEP[i] = next()
	}
	// Precompute XOR for makemove
	for i := 0; i < 16; i++ {
		for b := 0; b < 4; b++ {
			if i>>b&1 != 0 {
				zobristCastleDiff[i] ^= castleKeys[b]
			}
		}
	}
}

func initAttacks() {
	for sq := 0; sq < 64; sq++ {
		r, f := sq>>3, sq&7
		for to := 0; to < 64; to++ {
			dr, df := abs(to>>3-r), abs(to&7-f)
			if dr*df == 2 { // (1,2) or (2,1): a knight jump
				knightAttacks[sq] |= sqBB[to]
			}
			if max(dr, df) == 1 {
				kingAttacks[sq] |= sqBB[to]
			}
		}
		if r < 7 && f > 0 {
			pawnAttacks[White][sq] |= sqBB[sq+7]
		}
		if r < 7 && f < 7 {
			pawnAttacks[White][sq] |= sqBB[sq+9]
		}
		if r > 0 && f > 0 {
			pawnAttacks[Black][sq] |= sqBB[sq-9]
		}
		if r > 0 && f < 7 {
			pawnAttacks[Black][sq] |= sqBB[sq-7]
		}
	}
}

func popLSB(b *Bitboard) int {
	idx := bits.TrailingZeros64(uint64(*b))
	*b &= *b - 1
	return idx
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func newMove(from, to, flags int) Move {
	return Move(from | (to << 6) | (flags << 12))
}

func (m Move) from() int       { return int(m) & 63 }
func (m Move) to() int         { return int(m>>6) & 63 }
func (m Move) flags() int      { return int(m >> 12) }
func (m Move) isCapture() bool { return m.flags()&4 != 0 }
func (m Move) isPromo() bool   { return m.flags()&8 != 0 }
func (m Move) promoType() int  { return (m.flags() & 3) + Knight }

func (m Move) String() string {
	if m == 0 {
		return "0000"
	}
	var buf [5]byte
	from, to := m.from(), m.to()
	buf[0] = byte('a' + from%8)
	buf[1] = byte('1' + from/8)
	buf[2] = byte('a' + to%8)
	buf[3] = byte('1' + to/8)
	if m.isPromo() {
		buf[4] = "nbrq"[m.promoType()-Knight]
		return string(buf[:5])
	}
	return string(buf[:4])
}

func newPosition() *Position {
	p := &Position{}
	p.setStartPos()
	return p
}

func (p *Position) setStartPos() {
	p.setFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
}

// setFEN sets up the position from a FEN: piece placement, side to move, castling
// rights and en passant square, then the two optional clocks. It rejects positions
// the move generator cannot handle. On error p is left half-built, so set up a
// scratch Position when the current one must survive (uciLoop does).
func (p *Position) setFEN(fen string) error {
	parts := strings.Fields(fen)
	if len(parts) < 4 || len(parts) > 6 {
		return fmt.Errorf("expected 4 to 6 fields, got %d", len(parts))
	}
	*p = Position{}
	for i := range p.square {
		p.square[i] = -1
	}
	p.epSquare = -1

	if err := p.parseBoard(parts[0]); err != nil {
		return err
	}
	switch parts[1] {
	case "w":
	case "b":
		p.side = Black
		p.hash ^= zobristSide
	default:
		return fmt.Errorf("bad side to move %q", parts[1])
	}
	if p.isAttacked(p.kingSq[p.side^1], p.side, p.all) {
		return fmt.Errorf("the side not to move is in check")
	}
	if err := p.parseCastling(parts[2]); err != nil {
		return err
	}
	if err := p.parseEnPassant(parts[3]); err != nil {
		return err
	}

	// Halfmove clock (for the fifty-move rule) and fullmove number (unused)
	if len(parts) >= 5 {
		n, err := strconv.Atoi(parts[4])
		if err != nil || n < 0 {
			return fmt.Errorf("bad halfmove clock %q", parts[4])
		}
		p.halfmove = n
	}
	if len(parts) == 6 {
		if n, err := strconv.Atoi(parts[5]); err != nil || n < 0 {
			return fmt.Errorf("bad fullmove number %q", parts[5])
		}
	}

	p.historyKeys[0] = p.hash
	p.computePhase()
	p.nnRefresh(&p.acc[0], White)
	p.nnRefresh(&p.acc[0], Black)
	p.nn[0].ready = [2]bool{true, true}
	return nil
}

// parseBoard places the pieces of a FEN's first field, rank 8 first, and checks
// that the material could occur in a game.
func (p *Position) parseBoard(board string) error {
	ranks := strings.Split(board, "/")
	if len(ranks) != 8 {
		return fmt.Errorf("expected 8 ranks, got %d", len(ranks))
	}
	for r, rank := range ranks {
		file := 0
		for _, ch := range rank {
			if ch >= '1' && ch <= '8' {
				file += int(ch - '0')
				continue
			}
			i := strings.IndexRune("PNBRQKpnbrqk", ch)
			if i < 0 || file > 7 {
				return fmt.Errorf("bad rank %q", rank)
			}
			color, pt, sq := i/6, i%6, (7-r)*8+file
			file++
			bb := sqBB[sq]
			p.pieces[color][pt] |= bb
			p.occupied[color] |= bb
			p.all |= bb
			p.square[sq] = (color << 3) | pt
			p.hash ^= zobristPiece[color][pt][sq]
		}
		if file != 8 {
			return fmt.Errorf("bad rank %q", rank)
		}
	}
	for c := White; c <= Black; c++ {
		count := func(pt int) int { return bits.OnesCount64(uint64(p.pieces[c][pt])) }
		promoted := max(count(Knight)-2, 0) + max(count(Bishop)-2, 0) + max(count(Rook)-2, 0) + max(count(Queen)-1, 0)
		if count(King) != 1 || count(Pawn)+promoted > 8 {
			return fmt.Errorf("each side needs one king and at most 8 pawns and promoted pieces together")
		}
	}
	if (p.pieces[White][Pawn]|p.pieces[Black][Pawn])&0xFF000000000000FF != 0 {
		return fmt.Errorf("pawn on the first or last rank")
	}
	p.kingSq[White] = bits.TrailingZeros64(uint64(p.pieces[White][King]))
	p.kingSq[Black] = bits.TrailingZeros64(uint64(p.pieces[Black][King]))
	return nil
}

// parseCastling reads the castling rights; each needs its king and rook on their
// starting squares.
func (p *Position) parseCastling(rights string) error {
	if rights != "-" {
		for _, ch := range rights {
			i := strings.IndexRune("KQkq", ch)
			if i < 0 {
				return fmt.Errorf("bad castling rights %q", rights)
			}
			p.castle |= 1 << i
		}
	}
	for i, rookSq := range [4]int{7, 0, 63, 56} {
		c := i / 2
		if p.castle>>i&1 != 0 && (p.square[56*c+4] != c<<3|King || p.square[rookSq] != c<<3|Rook) {
			return fmt.Errorf("castling right %c without its king and rook at home", "KQkq"[i])
		}
	}
	p.hash ^= zobristCastleDiff[p.castle]
	return nil
}

// parseEnPassant reads the en passant square, which must be behind a pawn that
// just moved two squares.
func (p *Position) parseEnPassant(ep string) error {
	if ep == "-" {
		return nil
	}
	rank := 5 - 3*p.side // rank 6 with White to move, rank 3 with Black
	if len(ep) != 2 || ep[0] < 'a' || ep[0] > 'h' || int(ep[1])-'1' != rank {
		return fmt.Errorf("bad en passant square %q", ep)
	}
	p.epSquare = rank*8 + int(ep[0]-'a')
	if p.square[p.epSquare] != -1 || p.square[p.epSquare^8] != (p.side^1)<<3|Pawn {
		return fmt.Errorf("en passant square %s is not behind a pawn that just moved two squares", ep)
	}
	p.hash ^= zobristEP[p.epSquare%8]
	return nil
}

// setPosition applies the arguments of a UCI position command, "startpos" or
// "fen <fen>", optionally followed by "moves <moves>". On error p is left half-built.
func (p *Position) setPosition(args []string) error {
	movesAt := slices.Index(args, "moves")
	if movesAt < 0 {
		movesAt = len(args)
	}
	switch {
	case len(args) > 0 && args[0] == "startpos" && movesAt == 1:
		p.setStartPos()
	case len(args) > 0 && args[0] == "fen":
		if err := p.setFEN(strings.Join(args[1:movesAt], " ")); err != nil {
			return fmt.Errorf("invalid FEN: %w", err)
		}
	default:
		return fmt.Errorf("expected startpos or fen <fen>")
	}

	var buf [256]Move
	for _, s := range args[min(movesAt+1, len(args)):] {
		n, found := p.generateMovesTo(buf[:], false), false
		for _, m := range buf[:n] {
			if strings.EqualFold(m.String(), s) && p.isLegal(m) {
				p.makeMove(m)
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("illegal move %s", s)
		}
		if p.historyPly > MaxGamePly-1-MaxDepth {
			p.rebaseHistory()
		}
	}
	return nil
}

// rebaseHistory moves the end of the game history to the front of the history
// arrays, so a game can outgrow MaxGamePly with room left for a full-depth search.
// Repetition detection needs nothing before the last irreversible move, nor more
// than 100 plies: by then the fifty-move rule already scores the node as a draw.
func (p *Position) rebaseHistory() {
	keep := min(p.historyPly-p.lastIrreversible, 100)
	shift := p.historyPly - keep
	copy(p.historyKeys[:keep+1], p.historyKeys[shift:p.historyPly+1])
	p.historyPly = keep
	p.lastIrreversible = max(p.lastIrreversible-shift, 0)
	p.nnRefresh(&p.acc[keep], White)
	p.nnRefresh(&p.acc[keep], Black)
	p.nn[keep].ready = [2]bool{true, true}
}

func rookAttacks(sq int, occ Bitboard) Bitboard {
	m := &rookMagics[sq]
	idx := uint32(((occ & m.mask) * m.magic) >> m.shift)
	return rookAttackTable[m.offset+idx]
}

func bishopAttacks(sq int, occ Bitboard) Bitboard {
	m := &bishopMagics[sq]
	idx := uint32(((occ & m.mask) * m.magic) >> m.shift)
	return bishopAttackTable[m.offset+idx]
}

func (p *Position) isAttacked(sq, bySide int, occ Bitboard) bool {
	qu := p.pieces[bySide][Queen]
	return (pawnAttacks[bySide^1][sq]&p.pieces[bySide][Pawn] != 0) ||
		(knightAttacks[sq]&p.pieces[bySide][Knight] != 0) ||
		(kingAttacks[sq]&p.pieces[bySide][King] != 0) ||
		(bishopAttacks(sq, occ)&(p.pieces[bySide][Bishop]|qu) != 0) ||
		(rookAttacks(sq, occ)&(p.pieces[bySide][Rook]|qu) != 0)
}

func (p *Position) inCheck() bool {
	kingSq := p.kingSq[p.side]
	return p.isAttacked(kingSq, p.side^1, p.all)
}

/*
   Move Generation Strategy:
   Instead of looping over every square, we loop over pieces (Bitboards).

   Example:
   for bb := knights; bb != 0; {
       from := popLSB(&bb)         // Get location of a knight
       attacks := lookup(from)     // Get all target squares
       valid := attacks & ~us      // Remove friendly fire
   }

   This _Piece-Centric_ approach is much faster than _Square-Centric_ loops
   because empty squares are skipped entirely.
*/

func (p *Position) generateMovesTo(buf []Move, capturesOnly bool) int {
	i := 0
	us, them := p.side, p.side^1
	occAll, occUs, occThem, ep := p.all, p.occupied[us], p.occupied[them], p.epSquare
	push := 8 - 16*us // +8 for White, -8 for Black
	promoRank, dblRank := 6-5*us, 1+5*us

	for bb := p.pieces[us][Pawn]; bb != 0; {
		from := popLSB(&bb)
		to := from + push

		if from>>3 == promoRank {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				for flag := FlagPromoQ; flag >= FlagPromoN; flag-- {
					buf[i] = newMove(from, to, flag)
					i++
				}
			}
			for att := pawnAttacks[us][from] & occThem; att != 0; {
				to := popLSB(&att)
				for flag := FlagPromoCQ; flag >= FlagPromoCN; flag-- {
					buf[i] = newMove(from, to, flag)
					i++
				}
			}
		} else {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				buf[i] = newMove(from, to, FlagQuiet)
				i++
				if from>>3 == dblRank && occAll&sqBB[from+2*push] == 0 {
					buf[i] = newMove(from, from+2*push, FlagQuiet)
					i++
				}
			}
			for att := pawnAttacks[us][from] & occThem; att != 0; {
				buf[i] = newMove(from, popLSB(&att), FlagCapture)
				i++
			}
			if ep >= 0 && pawnAttacks[us][from]&sqBB[ep] != 0 {
				buf[i] = newMove(from, ep, FlagEP)
				i++
			}
		}
	}

	for pt := Knight; pt <= King; pt++ {
		for bb := p.pieces[us][pt]; bb != 0; {
			from := popLSB(&bb)
			var attacks Bitboard
			switch pt {
			case Knight:
				attacks = knightAttacks[from]
			case Bishop:
				attacks = bishopAttacks(from, occAll)
			case Rook:
				attacks = rookAttacks(from, occAll)
			case Queen:
				attacks = bishopAttacks(from, occAll) | rookAttacks(from, occAll)
			default:
				attacks = kingAttacks[from]
			}
			attacks &^= occUs
			if capturesOnly {
				attacks &= occThem
			}
			for attacks != 0 {
				to := popLSB(&attacks)
				flag := FlagQuiet
				if occThem&sqBB[to] != 0 {
					flag = FlagCapture
				}
				buf[i] = newMove(from, to, flag)
				i++
			}
		}
	}

	// castle bits: 1=K 2=Q 4=k 8=q; shift the side's pair down to bits 1 and 2
	if rights, base := p.castle>>(2*us), 56*us; !capturesOnly && rights&3 != 0 && !p.inCheck() {
		if rights&1 != 0 && occAll&(Bitboard(0x60)<<base) == 0 {
			buf[i] = newMove(base+4, base+6, FlagCastle)
			i++
		}
		if rights&2 != 0 && occAll&(Bitboard(0x0E)<<base) == 0 {
			buf[i] = newMove(base+4, base+2, FlagCastle)
			i++
		}
	}
	return i
}

func (p *Position) isLegal(m Move) bool {
	from, to := m.from(), m.to()
	us, them := p.side, p.side^1
	flags := m.flags()
	if flags == FlagCastle {
		d := 1
		if to < from {
			d = -1
		}
		return !p.isAttacked(from+d, them, p.all) && !p.isAttacked(from+2*d, them, p.all)
	}
	pt := p.square[from] & 7
	fromBB, toBB := sqBB[from], sqBB[to]

	capBB := Bitboard(0)
	if m.isCapture() {
		capSq := to
		if flags == FlagEP {
			capSq ^= 8
		}
		capBB = sqBB[capSq]
	}

	occ2 := (p.all &^ fromBB &^ capBB) | toBB
	kingSq := p.kingSq[us]
	if pt == King {
		kingSq = to
	}
	return !(pawnAttacks[them^1][kingSq]&(p.pieces[them][Pawn]&^capBB) != 0 || knightAttacks[kingSq]&(p.pieces[them][Knight]&^capBB) != 0 || bishopAttacks(kingSq, occ2)&((p.pieces[them][Bishop]|p.pieces[them][Queen])&^capBB) != 0 || rookAttacks(kingSq, occ2)&((p.pieces[them][Rook]|p.pieces[them][Queen])&^capBB) != 0 || kingAttacks[kingSq]&p.pieces[them][King] != 0)
}

func (p *Position) makeMove(m Move) Undo {
	undo := Undo{
		castle:           p.castle,
		epSquare:         p.epSquare,
		halfmove:         p.halfmove,
		captured:         -1,
		lastIrreversible: p.lastIrreversible,
	}

	from, to, flags := m.from(), m.to(), m.flags()
	us, them := p.side, p.side^1
	h := p.hash ^ zobristSide
	d := &p.nn[p.historyPly+1] // the accumulator change is recorded here, applied by nnBuild
	d.nAdd, d.nSub = 0, 0

	if p.epSquare >= 0 {
		h ^= zobristEP[p.epSquare%8]
		p.epSquare = -1
	}

	movingPiece := p.square[from] & 7

	if flags&FlagCapture != 0 {
		capSq := to
		if flags == FlagEP {
			capSq ^= 8
		}
		capturedPiece := p.square[capSq] & 7
		undo.captured = capturedPiece

		bb := sqBB[capSq]
		p.pieces[them][capturedPiece] &^= bb
		p.occupied[them] &^= bb
		p.all &^= bb
		h ^= zobristPiece[them][capturedPiece][capSq]
		p.phase += piecePhase[capturedPiece]
		d.removed(them, capturedPiece, capSq)
		p.square[capSq] = -1
	}

	if flags == FlagCastle {
		kingBB := sqBB[from] | sqBB[to]
		p.pieces[us][King] ^= kingBB
		p.occupied[us] ^= kingBB
		p.all ^= kingBB
		h ^= zobristPiece[us][King][from] ^ zobristPiece[us][King][to]
		p.square[from] = -1
		p.square[to] = (us << 3) | King
		p.kingSq[us] = to
		d.added(us, King, to)
		d.removed(us, King, from)
		rf, rt := from+3, from+1 // king side
		if to < from {
			rf, rt = from-4, from-1
		}
		rookBB := sqBB[rf] | sqBB[rt]
		p.pieces[us][Rook] ^= rookBB
		p.occupied[us] ^= rookBB
		p.all ^= rookBB
		h ^= zobristPiece[us][Rook][rf] ^ zobristPiece[us][Rook][rt]
		d.added(us, Rook, rt)
		d.removed(us, Rook, rf)
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		promoType := (flags & 3) + Knight
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][Pawn] ^= sqBB[from]
		p.pieces[us][promoType] ^= sqBB[to]
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		h ^= zobristPiece[us][Pawn][from] ^ zobristPiece[us][promoType][to]
		p.phase -= piecePhase[promoType]
		d.removed(us, Pawn, from)
		d.added(us, promoType, to)
		p.square[from] = -1
		p.square[to] = (us << 3) | promoType

	} else {
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][movingPiece] ^= moveBB
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		h ^= zobristPiece[us][movingPiece][from] ^ zobristPiece[us][movingPiece][to]
		p.square[from] = -1
		p.square[to] = (us << 3) | movingPiece
		if movingPiece == King {
			p.kingSq[us] = to
		}
		d.added(us, movingPiece, to)
		d.removed(us, movingPiece, from)

		if movingPiece == Pawn && abs(to-from) == 16 {
			p.epSquare = (from + to) / 2
			h ^= zobristEP[p.epSquare%8]
		}
	}

	p.castle &= castleMask[from] & castleMask[to]
	d.kingSq, d.ready = [2]int8{int8(p.kingSq[White]), int8(p.kingSq[Black])}, [2]bool{}
	// A king crossing files d/e flips its own view's mirroring: rebuild that view now,
	// while the board is at this ply. The other view stays lazy.
	if movingPiece == King && (from&7 < 4) != (to&7 < 4) {
		p.nnRefresh(&p.acc[p.historyPly+1], us)
		d.ready[us] = true
	}

	h ^= zobristCastleDiff[undo.castle^p.castle]
	p.side ^= 1
	p.historyPly++
	p.historyKeys[p.historyPly] = h
	p.halfmove++
	if flags&FlagCapture != 0 || movingPiece == Pawn {
		p.halfmove = 0
		p.lastIrreversible = p.historyPly
	}
	p.hash = h
	return undo
}

func (p *Position) unmakeMove(m Move, undo Undo) {
	from, to, flags := m.from(), m.to(), m.flags()
	us, them := p.side^1, p.side
	p.historyPly--
	p.lastIrreversible = undo.lastIrreversible
	p.side = us
	p.castle = undo.castle
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove

	if flags == FlagCastle {
		kingBB := sqBB[from] | sqBB[to]
		p.pieces[us][King] ^= kingBB
		p.occupied[us] ^= kingBB
		p.all ^= kingBB
		p.square[to] = -1
		p.square[from] = (us << 3) | King
		p.kingSq[us] = from

		rf, rt := from+1, from+3 // king side
		if to < from {
			rf, rt = from-1, from-4
		}

		rookBB := sqBB[rf] | sqBB[rt]
		p.pieces[us][Rook] ^= rookBB
		p.occupied[us] ^= rookBB
		p.all ^= rookBB
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		promoType := (flags & 3) + Knight
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][promoType] ^= sqBB[to]
		p.pieces[us][Pawn] ^= sqBB[from]
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		p.phase += piecePhase[promoType]
		p.square[to] = -1
		p.square[from] = (us << 3) | Pawn

	} else {
		movingPt := p.square[to] & 7
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][movingPt] ^= moveBB
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		p.square[to] = -1
		p.square[from] = (us << 3) | movingPt
		if movingPt == King {
			p.kingSq[us] = from
		}
	}

	if undo.captured >= 0 {
		capSq := to
		if flags == FlagEP {
			capSq ^= 8
		}
		bb := sqBB[capSq]
		capturedPiece := undo.captured
		p.pieces[them][capturedPiece] |= bb
		p.occupied[them] |= bb
		p.all |= bb
		p.square[capSq] = (them << 3) | capturedPiece
		p.phase -= piecePhase[capturedPiece]
	}
	p.hash = p.historyKeys[p.historyPly]
}

func (p *Position) makeNullMove() Undo {
	undo := Undo{
		epSquare: p.epSquare,
		halfmove: p.halfmove,
	}

	p.side ^= 1
	p.hash ^= zobristSide

	if epFile := p.epSquare; epFile != -1 {
		p.hash ^= zobristEP[epFile%8]
		p.epSquare = -1
	}
	p.halfmove++
	p.nn[p.historyPly+1] = NNPly{kingSq: [2]int8{int8(p.kingSq[White]), int8(p.kingSq[Black])}} // nothing changed
	p.historyPly++
	p.historyKeys[p.historyPly] = p.hash
	return undo
}

func (p *Position) unmakeNullMove(undo Undo) {
	p.historyPly--
	p.hash = p.historyKeys[p.historyPly]
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove
	p.side ^= 1
}

// see is the static exchange evaluation of m: the material balance after both
// sides keep recapturing on m's target square with their least valuable attacker.
func (p *Position) see(m Move) int {
	from, to := m.from(), m.to()
	var gain [32]int
	if m.flags() == FlagEP {
		gain[0] = pieceValues[Pawn]
	} else if p.square[to] != -1 {
		gain[0] = pieceValues[p.square[to]&7]
	}
	piece := p.square[from] & 7
	if m.isPromo() {
		piece = m.promoType()
	}
	diagSliders := p.pieces[White][Bishop] | p.pieces[Black][Bishop] | p.pieces[White][Queen] | p.pieces[Black][Queen]
	orthSliders := p.pieces[White][Rook] | p.pieces[Black][Rook] | p.pieces[White][Queen] | p.pieces[Black][Queen]
	occ := p.all &^ sqBB[from]
	att := pawnAttacks[White][to]&p.pieces[Black][Pawn] | pawnAttacks[Black][to]&p.pieces[White][Pawn] |
		knightAttacks[to]&(p.pieces[White][Knight]|p.pieces[Black][Knight]) |
		kingAttacks[to]&(p.pieces[White][King]|p.pieces[Black][King])
	// The moved piece leaves, which can uncover x-ray sliders behind it
	att = att&^sqBB[from] | p.getXrayAttackers(to, occ, diagSliders, orthSliders)
	us, d := p.side^1, 0
	for {
		myAtt := att & p.occupied[us]
		i := 0 // least valuable attacker first
		for i < len(lvaOrder) && myAtt&p.pieces[us][lvaOrder[i]] == 0 {
			i++
		}
		if i == len(lvaOrder) {
			break
		}
		d++
		gain[d] = pieceValues[piece] - gain[d-1]
		piece = lvaOrder[i]
		bb := sqBB[bits.TrailingZeros64(uint64(myAtt&p.pieces[us][piece]))]
		att &^= bb
		occ &^= bb
		att |= p.getXrayAttackers(to, occ, diagSliders, orthSliders)
		us ^= 1
	}
	for ; d > 0; d-- {
		gain[d-1] = -max(-gain[d-1], gain[d])
	}
	return gain[0]
}

func (p *Position) getXrayAttackers(sq int, occ, diagSliders, orthSliders Bitboard) Bitboard {
	return (bishopAttacks(sq, occ) & diagSliders & occ) | (rookAttacks(sq, occ) & orthSliders & occ)
}

func (s *Searcher) updateHistory(side, from, to, bonus int) {
	bonus = max(-HistoryBonusMax, min(HistoryBonusMax, bonus))
	s.history[side][from][to] += bonus - s.history[side][from][to]*abs(bonus)/MaxHistory
}

/*
  ----------------------------------------------------------------------------------
   NNUE EVALUATION
  ----------------------------------------------------------------------------------
   A small neural network trained on Soomi's own self-play games (nnue\DESIGN.md).

   board -> 768 inputs -> 128 sums x 2 views (SCReLU) -> 1 of 8 output buckets -> cp

   Inputs:  one on/off feature per (colour, piece, square) as seen by one side:
            own pieces 0-5, enemy pieces 6-11. Black's view is flipped top to bottom,
            and a view whose own king stands on files e-h is also flipped left to
            right, so every view sees its own king on files a-d.
   Sums:    128 per view (the accumulator), same weights for both views. makeMove only
            records which features changed; evaluate applies the pending changes
            forward from the last built ply, so nodes that never evaluate cost nothing.
            unmakeMove just steps historyPly back.
   Speed:   the column and output loops use AVX2 through simd/archsimd, so Soomi.go
            must be built with GOEXPERIMENT=simd. CPUs without AVX2 run scalar loops
            that give identical results.
   Output:  side to move's sums first, clamped to 0..1 and squared, mixed by the
            output layer of bucket (pieces on board - 2) / 4.
*/

const (
	NNInputs  = 768
	NNHidden  = 128
	NNBuckets = 8
	NNQA      = 255 // accumulator weights and biases are stored x255
	NNQB      = 64  // output weights are stored x64
)

//go:embed soomi.nnue
var netData []byte

var (
	nnFtW   [NNInputs * NNHidden]int16 // one NNHidden-wide column per feature
	nnFtB   [NNHidden]int16
	nnOutW  [NNBuckets][2 * NNHidden]int16
	nnOutB  [NNBuckets]int32
	nnScale int // S: maps the net's output to centipawns
	nnCRC   uint32
)

// loadNet parses a net file: a 20-byte header (magic "SNUE", then uint16 version,
// flags, inputs, hidden, buckets, QA, QB, S), the weights in the order declared
// above, all little-endian, and a CRC32 of everything before it.
func loadNet(b []byte) error {
	le := binary.LittleEndian
	size := 20 + 2*(len(nnFtW)+len(nnFtB)+NNBuckets*2*NNHidden) + 4*NNBuckets + 4
	if len(b) != size {
		return fmt.Errorf("net is %d bytes, expected %d", len(b), size)
	}
	if crc32.ChecksumIEEE(b[:size-4]) != le.Uint32(b[size-4:]) {
		return fmt.Errorf("checksum mismatch, the file is corrupt")
	}
	if string(b[:4]) != "SNUE" || le.Uint16(b[4:]) != 1 || le.Uint16(b[6:]) != 1 || le.Uint16(b[8:]) != NNInputs ||
		le.Uint16(b[10:]) != NNHidden || le.Uint16(b[12:]) != NNBuckets || le.Uint16(b[14:]) != NNQA || le.Uint16(b[16:]) != NNQB {
		return fmt.Errorf("header does not match this engine (768 -> 128x2 -> 8 buckets, mirrored)")
	}
	nnScale = int(le.Uint16(b[18:]))
	i := 20
	read := func(dst []int16) {
		for j := range dst {
			dst[j] = int16(le.Uint16(b[i:]))
			i += 2
		}
	}
	read(nnFtW[:])
	read(nnFtB[:])
	for k := range nnOutW {
		read(nnOutW[k][:])
		for _, x := range nnOutW[k] {
			if x < -128 || x > 128 { // the AVX2 eval needs 255*w to fit in int16
				return fmt.Errorf("output weight %d is outside -128..128", x)
			}
		}
	}
	for k := range nnOutB {
		nnOutB[k] = int32(le.Uint32(b[i:]))
		i += 4
	}
	nnCRC = le.Uint32(b[size-4:])
	return nil
}

// nnFeature is the input index of a colour-c piece pt on sq in view's accumulator;
// kingSq is view's own king.
func nnFeature(view, c, pt, sq, kingSq int) int {
	if kingSq&7 >= 4 {
		sq ^= 7
	}
	return ((c^view)*6+pt)*64 + (sq ^ 56*view)
}

// nnSIMD selects the AVX2 loops. Without AVX2 the scalar loops run; both give
// identical results.
var nnSIMD = archsimd.X86.AVX2()

func nnAddCol(a *[NNHidden]int16, f int) {
	w := nnFtW[f*NNHidden : f*NNHidden+NNHidden]
	if nnSIMD {
		for i := 0; i < NNHidden; i += 16 {
			archsimd.LoadInt16x16(a[i:]).Add(archsimd.LoadInt16x16(w[i:])).Store(a[i:])
		}
		return
	}
	for i := range a {
		a[i] += w[i]
	}
}

func nnSubCol(a *[NNHidden]int16, f int) {
	w := nnFtW[f*NNHidden : f*NNHidden+NNHidden]
	if nnSIMD {
		for i := 0; i < NNHidden; i += 16 {
			archsimd.LoadInt16x16(a[i:]).Sub(archsimd.LoadInt16x16(w[i:])).Store(a[i:])
		}
		return
	}
	for i := range a {
		a[i] -= w[i]
	}
}

// nnAddSubCol sets dst = src + column add - column sub; dst may be src.
func nnAddSubCol(dst, src *[NNHidden]int16, add, sub int) {
	wa, ws := nnFtW[add*NNHidden:add*NNHidden+NNHidden], nnFtW[sub*NNHidden:sub*NNHidden+NNHidden]
	if nnSIMD {
		for i := 0; i < NNHidden; i += 16 {
			archsimd.LoadInt16x16(src[i:]).Add(archsimd.LoadInt16x16(wa[i:])).Sub(archsimd.LoadInt16x16(ws[i:])).Store(dst[i:])
		}
		return
	}
	for i := range dst {
		dst[i] = src[i] + wa[i] - ws[i]
	}
}

// nnRefresh rebuilds view's half of acc from the board.
func (p *Position) nnRefresh(acc *[2][NNHidden]int16, view int) {
	acc[view] = nnFtB
	for c := White; c <= Black; c++ {
		for pt := Pawn; pt <= King; pt++ {
			for bb := p.pieces[c][pt]; bb != 0; {
				nnAddCol(&acc[view], nnFeature(view, c, pt, popLSB(&bb), p.kingSq[view]))
			}
		}
	}
}

// NNPly is what makeMove records instead of updating the accumulator: the pieces
// added and removed, packed as (colour*6+piece)<<6 | square, both kings (they decide
// each view's mirroring), and which views of acc at this ply are already built.
// A null move adds and removes nothing.
type NNPly struct {
	add, sub   [2]int16
	nAdd, nSub int8
	kingSq     [2]int8
	ready      [2]bool
}

func (d *NNPly) added(c, pt, sq int) {
	d.add[d.nAdd] = int16((c*6+pt)<<6 | sq)
	d.nAdd++
}

func (d *NNPly) removed(c, pt, sq int) {
	d.sub[d.nSub] = int16((c*6+pt)<<6 | sq)
	d.nSub++
}

// nnFeat is nnFeature for a piece packed by NNPly.
func nnFeat(view int, piece int16, kingSq int8) int {
	return nnFeature(view, int(piece>>6)/6, int(piece>>6)%6, int(piece&63), int(kingSq))
}

// nnBuild brings both views of the accumulator at historyPly up to date: it applies
// the recorded changes forward from the nearest built ply (setFEN builds ply 0).
func (p *Position) nnBuild() {
	for v := White; v <= Black; v++ {
		k := p.historyPly
		for !p.nn[k].ready[v] {
			k--
		}
		for k++; k <= p.historyPly; k++ {
			d, dst := &p.nn[k], &p.acc[k][v]
			if d.nAdd == 0 { // null move
				*dst = p.acc[k-1][v]
			} else {
				nnAddSubCol(dst, &p.acc[k-1][v], nnFeat(v, d.add[0], d.kingSq[v]), nnFeat(v, d.sub[0], d.kingSq[v]))
				if d.nAdd == 2 { // castling: the rook moved too
					nnAddSubCol(dst, dst, nnFeat(v, d.add[1], d.kingSq[v]), nnFeat(v, d.sub[1], d.kingSq[v]))
				} else if d.nSub == 2 { // a capture
					nnSubCol(dst, nnFeat(v, d.sub[1], d.kingSq[v]))
				}
			}
			d.ready[v] = true
		}
	}
}

// evaluate scores the position for the side to move, in centipawns.
func (p *Position) evaluate() int {
	p.nnBuild()
	acc := &p.acc[p.historyPly]
	b := min((bits.OnesCount64(uint64(p.all))-2)/4, NNBuckets-1) // an illegal FEN can hold more than 32 pieces
	sum := 0
	if nnSIMD {
		// Clamp to 0..255 and multiply by the weight (fits int16: 255*128), then
		// VPMADDWD multiplies by the clamped value again and sums pairs into int32.
		zero, top, s := archsimd.BroadcastInt16x16(0), archsimd.BroadcastInt16x16(NNQA), archsimd.BroadcastInt32x8(0)
		for half, view := range [2]int{p.side, p.side ^ 1} {
			a, w := acc[view][:], nnOutW[b][half*NNHidden:]
			for i := 0; i < NNHidden; i += 16 {
				c := archsimd.LoadInt16x16(a[i:]).Max(zero).Min(top)
				s = s.Add(c.DotProductPairs(c.Mul(archsimd.LoadInt16x16(w[i:]))))
			}
		}
		var lanes [8]int32
		s.Store(lanes[:])
		for _, x := range lanes {
			sum += int(x)
		}
	} else {
		us, them, w := &acc[p.side], &acc[p.side^1], &nnOutW[b]
		for i := range us {
			u, t := int(min(max(us[i], 0), NNQA)), int(min(max(them[i], 0), NNQA))
			sum += u*u*int(w[i]) + t*t*int(w[NNHidden+i])
		}
	}
	cp := (sum/NNQA + int(nnOutB[b])) * nnScale / (NNQA * NNQB)
	return max(-(Mate - MateScoreGuard - 1), min(Mate-MateScoreGuard-1, cp))
}

/*
  ----------------------------------------------------------------------------------
   MOVE ORDERING
  ----------------------------------------------------------------------------------
   To prune the search tree effectively, we must search the best moves first.

   Sorting Priority List:
   1. Hash Move       --> The best move stored in the transposition table
   2. Promotions      --> By the value of the promoted piece
   3. Good Captures   --> Captures that do not lose material, by SEE and MVV-LVA
   4. Killer Moves    --> Quiet moves that caused a cutoff at the same ply
   5. Countermoves    --> The quiet move that last refuted the opponent's previous move
   6. History/PST     --> Other quiet moves, by history plus a PST delta
   7. Losing Captures --> Negative SEE, after the quiet moves
*/

func (s *Searcher) clearHeuristics() {
	s.history = [2][64][64]int{}
	s.countermoves = [2][64][64]Move{}
}

func (s *Searcher) orderMoves(moves []Move, bestMove Move, killer1, killer2 Move, prevMove Move) []Move {
	p := &s.pos
	var stackScores [256]int
	scores := stackScores[:len(moves)]
	for i, m := range moves {
		switch {
		case m == bestMove:
			scores[i] = ScoreHash
		case m.isPromo() || m.isCapture():
			scores[i] = p.scoreNoisy(m)
		case m == killer1:
			scores[i] = ScoreKiller1
		case m == killer2:
			scores[i] = ScoreKiller2
		default:
			from, to := m.from(), m.to()
			pt := p.square[from] & 7
			// History, plus a PST delta for moves without history
			scores[i] = s.history[p.side][from][to] + pst[p.side][pt][to] - pst[p.side][pt][from]
			if prevMove != 0 && m == s.countermoves[p.side][prevMove.from()][prevMove.to()] {
				scores[i] += ScoreCountermove
			}
		}
	}
	sortByScore(moves, scores)
	return moves
}

func (p *Position) orderMovesQ(moves []Move, scores []int) {
	for i, m := range moves {
		scores[i] = 0
		if m.isPromo() || m.isCapture() {
			scores[i] = p.scoreNoisy(m)
		}
	}
	sortByScore(moves, scores)
}

// scoreNoisy ranks promotions by piece, then SEE-safe captures by SEE and
// MVV-LVA. Losing captures keep their negative SEE so they sort last.
func (p *Position) scoreNoisy(m Move) int {
	if m.isPromo() {
		return ScorePromoBase + pieceValues[m.promoType()]
	}
	seeVal := p.see(m)
	if seeVal < 0 {
		return seeVal
	}
	victim := Pawn
	if vt := p.square[m.to()]; vt != -1 {
		victim = vt & 7
	}
	return ScoreCaptureBase + seeVal + pieceValues[victim]*MVVLVAWeight - pieceValues[p.square[m.from()]&7]
}

// sortByScore is a descending insertion sort. It is stable: equal scores keep
// generation order, which the search's node counts depend on.
func sortByScore(moves []Move, scores []int) {
	for i := 1; i < len(moves); i++ {
		m, s := moves[i], scores[i]
		j := i - 1
		for ; j >= 0 && scores[j] < s; j-- {
			moves[j+1], scores[j+1] = moves[j], scores[j]
		}
		moves[j+1], scores[j+1] = m, s
	}
}

/*
  ----------------------------------------------------------------------------------
   QUIESCENCE SEARCH
  ----------------------------------------------------------------------------------
   Standard search stops at a fixed depth, but this can lead to the "Horizon Effect",
   where the engine misses critical tactical sequences just beyond the search depth.

   Scenario:
   Depth 4: White takes Black Queen. Eval says White is +900. STOP (time ran out etc...)
   Depth 5: Black takes back White Queen immediately after. Eval is actually Equal.

   Solution:
   When Depth is 0, do NOT stop if there are "noisy" moves (usually limited to captures), though some people include checks and promotions.
   Keep searching strictly through noisy moves until the position is "Quiet".
*/

func (s *Searcher) quiesce(alpha, beta, ply int) int {
	p := &s.pos
	s.seldepth = max(s.seldepth, ply)
	if ply >= MaxDepth {
		return p.evaluate()
	}
	s.nodes++
	if s.nodes&NodeCheckMaskSearch == 0 && s.tc.shouldStop() {
		return alpha
	}

	inCheck := p.inCheck()
	best := alpha
	if !inCheck {
		stand := p.evaluate()
		if stand >= beta {
			return stand
		}
		best, alpha = stand, max(alpha, stand)
		if !p.isEndgame() {
			them := p.side ^ 1
			maxGain := 0
			if p.pieces[them][Queen] != 0 {
				maxGain = pieceValues[Queen]
			} else if p.pieces[them][Rook] != 0 {
				maxGain = pieceValues[Rook]
			} else if (p.pieces[them][Bishop] | p.pieces[them][Knight]) != 0 {
				maxGain = pieceValues[Bishop]
			} else if p.pieces[them][Pawn] != 0 {
				maxGain = pieceValues[Pawn]
			}
			if stand+maxGain+DeltaMargin < alpha {
				return stand
			}
		}
	}

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], !inCheck)
	moves := movesArr[:n]
	var stackScores [256]int
	scores := stackScores[:n]
	p.orderMovesQ(moves, scores)

	legalCount := 0
	for i, m := range moves {
		if s.tc.shouldStop() {
			return alpha
		}
		// Prune losing captures, but never check evasions
		if !inCheck && scores[i] < 0 {
			continue
		}
		if !p.isLegal(m) {
			continue
		}
		legalCount++

		undo := p.makeMove(m)
		score := -s.quiesce(-beta, -alpha, ply+1)
		p.unmakeMove(m, undo)

		if score >= beta {
			return beta
		}
		best, alpha = max(best, score), max(alpha, score)
	}

	if inCheck && legalCount == 0 {
		return -Mate + ply
	}

	return best
}

/*
  ----------------------------------------------------------------------------------
   NEGAMAX SEARCH WITH ALPHA-BETA PRUNING
  ----------------------------------------------------------------------------------
   This function explores the game tree to find the best move. It uses the Negamax
   framework where max(a, b) = -min(-b, -a), simplifying code for 2-player zero-sum games.

          [ Root Node ]
          /     |     \
      [Move A] [Move B] [Move C]
        /         |        \
     ...         ...      (Pruned?)

   Techniques used here, in the order they run:
   1. Check extension, then the Transposition Table (TT): reuse results of positions already searched.
   2. Internal Iterative Reduction (IIR) and mate distance pruning.
   3. Reverse Futility Pruning (RFP), Null Move Pruning (NMP) and ProbCut: cut nodes whose
      static eval, a null move or a reduced search is already far enough above beta.
   4. Late Move Pruning (LMP) and SEE pruning: skip late quiet moves and losing moves at low depth.
   5. Principal Variation Search (PVS) with Late Move Reductions (LMR).
   6. Quiescence Search: At leaf nodes, play out captures to avoid "horizon effects".

   Alpha (α): Best score the maximizing player can guarantee so far.
   Beta  (β): Best score the minimizing player can guarantee so far.
   Condition: If Score >= Beta, we have a "Cutoff" (branch is too good, opponent won't allow it).
*/

func (s *Searcher) negamax(depth, alpha, beta, ply int, pvNode bool, prevMove Move) int {
	p := &s.pos
	s.seldepth = max(s.seldepth, ply)
	if ply >= MaxDepth {
		return p.evaluate()
	}
	s.nodes++
	if s.nodes&NodeCheckMaskSearch == 0 && s.tc.shouldStop() { // time check
		return alpha
	}

	inCheck := p.inCheck()
	if inCheck {
		depth++
	}
	if depth <= 0 {
		return s.quiesce(alpha, beta, ply)
	}

	origAlpha := alpha
	hashMove, ttScore, ttCutoff := s.probeTT(depth, alpha, beta, ply, pvNode)
	if ttCutoff {
		return ttScore
	}

	// IIR
	if depth >= IIRDepthMin && hashMove == 0 {
		depth--
	}

	// Mate distance pruning
	if beta > Mate-ply {
		if alpha >= Mate-ply {
			return Mate - ply
		}
		beta = Mate - ply
	}
	if alpha < -Mate+ply {
		if -Mate+ply >= beta {
			return -Mate + ply
		}
		alpha = -Mate + ply
	}

	if !inCheck {
		if score, cutoff := s.pruneNode(depth, beta, ply, prevMove); cutoff {
			return score
		}
	}

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	moves := s.orderMoves(movesArr[:n], hashMove, s.ss[ply].killer1, s.ss[ply].killer2, prevMove)

	bestMove := Move(0)
	bestScore := -Infinity
	legalMoves := 0
	var quietsTried [256]Move
	quietCount := 0

	for _, m := range moves {
		if !p.isLegal(m) {
			continue
		}
		legalMoves++
		if ply == 0 && depth > 4 && time.Since(s.start) >= 500*time.Millisecond {
			s.out.printf("info depth %d currmove %v currmovenumber %d\n", depth, m, legalMoves)
		}
		isQuiet := !m.isCapture() && !m.isPromo()
		if !inCheck && m != hashMove && bestScore > -Mate+MaxDepth && s.prunesMove(m, depth, legalMoves, quietCount, isQuiet) {
			continue
		}
		if isQuiet {
			quietsTried[quietCount] = m
			quietCount++
		}

		score := s.searchMove(m, depth, alpha, beta, ply, legalMoves, pvNode, inCheck, isQuiet)
		// A stopped child returns a meaningless score: store and learn nothing from it
		if s.tc.shouldStop() {
			return alpha
		}

		if score >= beta {
			if isQuiet && m != hashMove {
				s.updateQuietStats(m, depth, ply, prevMove, quietsTried[:quietCount-1])
			}
			tt.save(p.hash, m, scoreToTT(score, ply), depth, TTFlagLower)
			return score
		}
		if score > bestScore {
			bestScore, bestMove = score, m
		}
		if score > alpha {
			alpha = score
			if pvNode {
				n := copy(s.ss[ply].pv[1:], s.ss[ply+1].pv[:s.ss[ply+1].pvLen])
				s.ss[ply].pv[0], s.ss[ply].pvLen = m, n+1
			}
		}
	}

	// Handle draw & checkmate results
	if legalMoves == 0 {
		score := 0
		if inCheck {
			score = -Mate + ply
		}
		tt.save(p.hash, 0, scoreToTT(score, ply), depth, TTFlagExact)
		return score
	}

	flag := TTFlagExact
	if bestScore <= origAlpha {
		flag = TTFlagUpper
	}
	tt.save(p.hash, bestMove, scoreToTT(bestScore, ply), depth, flag)
	return bestScore
}

// probeTT looks the node up in the transposition table. It returns the stored move
// and, at non-PV nodes, whether the stored score and bound end the search here.
func (s *Searcher) probeTT(depth, alpha, beta, ply int, pvNode bool) (Move, int, bool) {
	move, score, flag, found, usable := tt.probe(s.pos.hash, depth)
	if !found || pvNode {
		return move, 0, false
	}
	// Mate scores are stored relative to this node and are usable at any depth
	if score > Mate-MateScoreGuard {
		score, usable = score-ply, true
	} else if score < -Mate+MateScoreGuard {
		score, usable = score+ply, true
	}
	cutoff := usable && (flag == TTFlagExact || (flag == TTFlagLower && score >= beta) || (flag == TTFlagUpper && score <= alpha))
	return move, score, cutoff
}

// pruneNode tries to cut a node that is not in check before its moves are searched:
// RFP when the static eval is far above beta, then a null move search, then ProbCut.
func (s *Searcher) pruneNode(depth, beta, ply int, prevMove Move) (int, bool) {
	p := &s.pos
	if p.isEndgame() {
		return 0, false
	}
	if depth <= RFPDepthMax {
		if eval := p.evaluate(); eval >= beta+RFPMargin*depth {
			return eval, true // soft fail
		}
	}
	if depth >= NMPDepthMin {
		R := NMPReductionBase + depth/NMPReductionDiv
		undo := p.makeNullMove()
		score := -s.negamax(depth-1-R, -beta, -beta+1, ply+1, false, 0)
		p.unmakeNullMove(undo)
		if score >= beta {
			return beta, true
		}
	}
	if probBeta := beta + ProbCutMargin; depth >= ProbCutDepthMin && probBeta <= Mate-MateScoreGuard {
		if score := s.negamax(depth-ProbCutReduction, probBeta-1, probBeta, ply+1, false, prevMove); score >= probBeta {
			return score, true // soft fail
		}
	}
	return 0, false
}

// prunesMove reports whether LMP or SEE pruning skips move m. The caller has
// checked that the node is not in check, m is not the hash move and a move scored.
func (s *Searcher) prunesMove(m Move, depth, legalMoves, quietCount int, isQuiet bool) bool {
	// LMP: at shallow depth, stop searching quiets once enough were tried
	if isQuiet && depth <= LMPDepthMax && quietCount >= LMPBase+depth*depth {
		return true
	}
	// SEE pruning: skip shallow moves that lose material
	if depth > SEEPruneDepthMax || legalMoves <= 1 || m.isPromo() {
		return false
	}
	seeMargin := -SEENoisyCoeff * depth * depth
	if isQuiet {
		seeMargin = -SEEQuietCoeff * depth
	}
	return s.pos.see(m) < seeMargin
}

// searchMove plays m, searches it and takes it back. A draw scores exactly 0.
// Otherwise late quiet moves get a reduced zero-window probe first (LMR), later moves
// at PV nodes a zero-window probe (PVS), and a full re-search only if they beat alpha.
func (s *Searcher) searchMove(m Move, depth, alpha, beta, ply, legalMoves int, pvNode, inCheck, isQuiet bool) int {
	p := &s.pos
	undo := p.makeMove(m)
	s.ss[ply+1].pvLen = 0
	score := 0
	if p.halfmove < 100 && !p.isRepetition() && !p.isInsufficientMaterial() {
		childDepth := depth - 1
		canReduce := childDepth >= LMRMinChildDepth && !inCheck && isQuiet && legalMoves > LMRLateMoveAfter
		zeroWindow := canReduce || (legalMoves > 1 && pvNode)
		if zeroWindow {
			d := childDepth
			if canReduce {
				red := lmrTable[min(depth, MaxDepth)][min(legalMoves, 255)]
				// Reduce worse lines more
				if !pvNode {
					red++
				}
				if s.history[p.side^1][m.from()][m.to()] < 0 {
					red++
				}
				d = max(1, childDepth-red)
			}
			score = -s.negamax(d, -alpha-1, -alpha, ply+1, false, m)
		}
		if !zeroWindow || score > alpha {
			score = -s.negamax(childDepth, -beta, -alpha, ply+1, pvNode, m)
		}
	}
	p.unmakeMove(m, undo)
	return score
}

// updateQuietStats rewards the quiet move m that caused a cutoff (killers, history
// bonus, countermove) and gives a history malus to the quiet moves tried before it.
func (s *Searcher) updateQuietStats(m Move, depth, ply int, prevMove Move, tried []Move) {
	side := s.pos.side
	k := &s.ss[ply]
	if m != k.killer1 {
		k.killer2, k.killer1 = k.killer1, m
	}
	bonus := depth * depth
	s.updateHistory(side, m.from(), m.to(), bonus)
	for _, q := range tried {
		s.updateHistory(side, q.from(), q.to(), -bonus)
	}
	s.countermoves[side][prevMove.from()][prevMove.to()] = m
}

/*
  ----------------------------------------------------------------------------------
   ITERATIVE DEEPENING SEARCH
  ----------------------------------------------------------------------------------
   Instead of searching directly to Depth 10, we search Depth 1, then 2, then 3...
   This might seem slower than just going directly to depth 10, but it offers unique advantages:

   Visual
   [Start]
    Search D=1 -> BestMove A
    Search D=2 -> BestMove A (uses info from depth 1 to sort moves)
    Search D=3 -> BestMove B (found a better move)
    [Time Up!] -> The unfinished iteration is thrown away; return the best move of the
                  last completed one.

   Depth 1 always completes: it ignores stop and the deadline (it takes well under a
   millisecond), so with a legal move on the board there is always a searched move to report.

   Benefits:
   1. Time Management: We always have a "best move so far" if we must stop abruptly (as we do in chess).
   2. Move Ordering: The BestMove from Depth X-1 is the first move searched at Depth X.
*/

// printInfo emits one UCI info line. bound is "", "lowerbound" or "upperbound".
func (s *Searcher) printInfo(depth, score int, pv []Move, elapsed time.Duration, bound string) {
	nps := int64(0)
	if elapsed > 0 {
		nps = int64(float64(s.nodes) / elapsed.Seconds())
	}
	scoreStr := fmt.Sprintf("cp %d", score)
	if abs(score) >= Mate-MateScoreGuard {
		// Mate distance in moves, signed from our point of view
		mateMoves := (Mate - abs(score) + 1) / 2
		if score < 0 {
			mateMoves = -mateMoves
		}
		scoreStr = fmt.Sprintf("mate %d", mateMoves)
	}
	if bound != "" {
		scoreStr += " " + bound
	}
	// Built whole and written once, so no other output can land inside the line
	var line strings.Builder
	fmt.Fprintf(&line, "info depth %d seldepth %d score %s nodes %d time %d nps %d hashfull %d",
		depth, s.seldepth, scoreStr, s.nodes, elapsed.Milliseconds(), nps, tt.hashfull())
	if len(pv) > 0 {
		line.WriteString(" pv")
		for _, m := range pv {
			line.WriteString(" " + m.String())
		}
	}
	s.out.printf("%s\n", line.String())
}

func (s *Searcher) search(tc *TimeControl) Move {
	var bestMove Move
	s.ss = [MaxDepth + 1]SearchStack{}

	maxDepth := tc.depth
	if maxDepth <= 0 || tc.infinite {
		maxDepth = MaxDepth
	}

	// Start timers
	s.nodes = 0
	s.seldepth = 0
	start := time.Now()
	s.start = start
	var prevScore int
	var prevBestMove Move
	stableIterations := 0
	lastIterElapsed := time.Duration(0)
	for depth := 1; depth <= maxDepth; depth++ {
		s.tc = tc
		if depth == 1 {
			s.tc = &TimeControl{} // never stops, see above
		}
		s.seldepth = depth
		s.ss[0].pvLen = 0
		var score int

		// Aspiration windows
		if depth >= AspirationStartDepth {
			window := AspirationBase
			low, high := prevScore-window, prevScore+window
			for {
				low, high = max(low, -Infinity), min(high, Infinity)
				score = s.negamax(depth, low, high, 0, true, 0)

				if tc.shouldStop() {
					break
				}

				if score <= low {
					// Failed low: true score is at most this
					s.printInfo(depth, score, s.ss[0].pv[:s.ss[0].pvLen], time.Since(start), "upperbound")
					low -= window
					window *= 2
				} else if score >= high {
					// Failed high: true score is at least this
					s.printInfo(depth, score, s.ss[0].pv[:s.ss[0].pvLen], time.Since(start), "lowerbound")
					high += window
					window *= 2
				} else {
					break
				}

				if window >= AspirationMaxWindow {
					score = s.negamax(depth, -Infinity, Infinity, 0, true, 0)
					break
				}
			}
		} else {
			score = s.negamax(depth, -Infinity, Infinity, 0, true, 0)
		}
		elapsed := time.Since(start)

		if depth > 1 && tc.shouldStop() {
			break
		}

		pv := s.ss[0].pv[:s.ss[0].pvLen]
		if len(pv) > 0 {
			bestMove = pv[0]
			if depth > 1 {
				if bestMove == prevBestMove {
					stableIterations++
				} else {
					stableIterations = 0
				}
			}
			prevBestMove = bestMove
		}

		// Print search info
		s.printInfo(depth, score, pv, elapsed, "")

		// Stability and score drop heuristics
		scale := 1.0
		if depth >= TMScaleMinDepth {
			if stableIterations >= TMStableIterations {
				// Best move has remained stable: save time!
				scale *= TMStableScale
			} else if stableIterations == 0 {
				// Best move changed this iteration: invest more time!
				scale *= TMUnstableScale
			}

			if prevScore != 0 && score < prevScore-TMScoreDrop {
				// Unexpected score drop: think harder to find defense
				scale *= TMScoreDropScale
			}
		}
		prevScore = score

		iterTime := elapsed - lastIterElapsed
		lastIterElapsed = elapsed

		if !tc.shouldContinue(elapsed, iterTime, scale) {
			break
		}
	}
	return bestMove
}

/*
  ----------------------------------------------------------------------------------
   TIME MANAGEMENT
  ----------------------------------------------------------------------------------
   Deciding how much time to spend on a move is a hard balance.
   - Too little: We play hasty, weak moves.
   - Too much: We likely flag (run out of time) later in the game.

   Each move gets two limits, after keeping the Move Overhead back for GUI and pipe lag
   (go movetime is used as given):
   1. Soft limit (optimumMs): RemainingTime / MovesToGo + TMIncrementPct% of the increment,
      MovesToGo defaulting to DefaultMovesToGo. Iterative deepening stops between
      iterations once past it, scaled by best move stability and score drops.
   2. Hard limit (deadline): TMHardLimitPct% of the soft limit, never more than the time
      left. The search stops mid-iteration when it is reached.
*/

func (tc *TimeControl) stop() {
	atomic.StoreInt32(&tc.stopped, 1)
}

func (tc *TimeControl) allocateTime(side int, overheadMs int64) {
	if tc.infinite || tc.depth > 0 {
		return
	}
	if tc.movetime > 0 {
		tc.optimumMs = tc.movetime
		tc.deadline = time.Now().Add(time.Duration(tc.movetime) * time.Millisecond)
		return
	}

	t, i, mtg := tc.wtime, tc.winc, int64(tc.movestogo)
	if side == Black {
		t, i = tc.btime, tc.binc
	}
	if mtg <= 0 {
		mtg = DefaultMovesToGo
	}

	avail := max(t-overheadMs, 0)
	// Soft target: fair fraction of remaining time plus part of the increment.
	// With nothing available both limits bottom out at MinTimeMs, so no special case.
	optimum := max(min(avail/mtg+i*TMIncrementPct/100, avail), MinTimeMs)
	// Hard emergency ceiling: a multiple of the soft target, strictly capped at available time
	maxTime := max(min(optimum*TMHardLimitPct/100, avail), MinTimeMs)

	tc.optimumMs = optimum
	tc.deadline = time.Now().Add(time.Duration(maxTime) * time.Millisecond)
}

func (tc *TimeControl) shouldStop() bool {
	// deadline is unset or built from time.Now(), so comparing with time.Time{} equals IsZero here
	// and keeps this inlinable
	return atomic.LoadInt32(&tc.stopped) != 0 || (tc.deadline != time.Time{} && time.Until(tc.deadline) <= 0)
}

func (tc *TimeControl) shouldContinue(elapsed, iterTime time.Duration, scale float64) bool {
	if atomic.LoadInt32(&tc.stopped) != 0 {
		return false
	}

	if tc.infinite || tc.depth > 0 || tc.movetime > 0 || tc.optimumMs <= 0 {
		return true
	}

	softTarget := time.Duration(float64(tc.optimumMs)*scale) * time.Millisecond

	// 1. Clean stop if soft time budget has been reached
	if elapsed >= softTarget {
		return false
	}

	// 2. Safety check against emergency hard deadline
	remainHard := time.Until(tc.deadline)
	if remainHard <= ContinueMargin {
		return false
	}

	// 3. Project next iteration time from the completed iteration
	if next := iterTime * TMNextIterationFactor; iterTime > 0 && (elapsed+next > softTarget*TMSoftOverrunPct/100 || next+ContinueMargin > remainHard) {
		return false
	}

	return true
}

/*
  ----------------------------------------------------------------------------------
   PERFT (Performance Test & Move Generation Validator)
  ----------------------------------------------------------------------------------
   Perft is a debugging function that traverses the move tree to a specific depth
   and counts the number of leaf nodes.

   Why is this useful?
   It verifies that the move generator (generateMovesTo, makeMove, unmakeMove) is
   mostly bug-free. We compare the results against known values for the start position.

   Example (Depth 1):
   Start Pos -> 20 legal moves. Perft(1) should return 20.

   Divide:
   "Divide" prints the child count for *each* root move separately.
   Perft 2 Divide Example:
   e2e4: 20
   e2e3: 20
   g1f3: 20
   --------
   Total: 400
   This isolates exactly which move branch contains a possible bug, if node counts differ from known results.
*/

func (p *Position) perft(depth int) int {
	if depth == 0 {
		return 1
	}
	var moves [256]Move
	count := 0
	for _, m := range moves[:p.generateMovesTo(moves[:], false)] {
		if p.isLegal(m) {
			undo := p.makeMove(m)
			count += p.perft(depth - 1)
			p.unmakeMove(m, undo)
		}
	}
	return count
}

func (p *Position) perftDivide(depth int, out *uciWriter) {
	var moves [256]Move
	total := 0
	for _, m := range moves[:p.generateMovesTo(moves[:], false)] {
		if p.isLegal(m) {
			undo := p.makeMove(m)
			count := p.perft(depth - 1)
			p.unmakeMove(m, undo)
			out.printf("%v: %d\n", m, count)
			total += count
		}
	}
	out.printf("\nTotal: %d\n", total)
}

/*
  ----------------------------------------------------------------------------------
   UCI MAIN LOOP (Universal Chess Interface)
  ----------------------------------------------------------------------------------
   This is the communication part. The GUI (Arena, Banksia, Cutechess) sends text commands, we reply with text.

   [ GUI ] -------- "position startpos moves e2e4" ---->  [ ENGINE ]
   [ GUI ] <------- "info depth 5 score cp 20..." ------  [ ENGINE ]
   [ GUI ] -------- "go wtime 60000" ------------------>  [ ENGINE ]
   [ GUI ] <------- "bestmove e7e5" --------------------  [ ENGINE ]

   The loop reads one command at a time. A search runs in its own goroutine, so stop
   and isready are answered while it thinks. Every go gets exactly one bestmove, and
   go infinite gets it only after stop. Bad input is reported with "info string" and
   changes nothing.
*/

// uciWriter serialises everything the engine prints: the UCI loop and the search
// goroutine both write, and every line has to reach the GUI whole.
type uciWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (u *uciWriter) printf(format string, a ...any) {
	u.mu.Lock()
	defer u.mu.Unlock()
	fmt.Fprintf(u.w, format, a...)
}

// uci is one UCI session: the GUI's position, the searcher that thinks about it and
// the search running in the background, if any. Only the loop goroutine uses it,
// apart from the search goroutine's own searcher and output.
type uci struct {
	out          *uciWriter
	pos          *Position
	scratch      *Position // position commands are set up here, then swapped in
	searcher     *Searcher
	moveOverhead int64
	current      *TimeControl // the latest search, possibly finished
	searchWG     sync.WaitGroup
}

func uciLoop(in io.Reader, out io.Writer) {
	u := &uci{
		out:          &uciWriter{w: out},
		pos:          newPosition(),
		scratch:      new(Position),
		searcher:     new(Searcher),
		moveOverhead: DefaultMoveOverheadMs,
	}
	u.searcher.out = u.out
	scanner := bufio.NewScanner(in)
	for scanner.Scan() {
		if parts := strings.Fields(scanner.Text()); len(parts) > 0 && !u.handle(parts) {
			break
		}
	}
	u.stopSearch()
}

// handle runs one command and reports whether to keep reading.
func (u *uci) handle(parts []string) bool {
	switch parts[0] {
	case "uci":
		u.out.printf("id name %s\nid author Otto Laukkanen\n", EngineName)
		u.out.printf("option name Hash type spin default %d min 1 max 4096\n", DefaultTTSizeMB)
		u.out.printf("option name Move Overhead type spin default %d min 0 max 5000\n", DefaultMoveOverheadMs)
		u.out.printf("uciok\n")
	case "isready":
		u.out.printf("readyok\n")
	case "setoption":
		u.setOption(parts)
	case "ucinewgame":
		u.stopSearch()
		tt.newGeneration()
		u.searcher.clearHeuristics()
		u.pos.setStartPos()
	case "position":
		u.stopSearch()
		if err := u.scratch.setPosition(parts[1:]); err != nil {
			u.out.printf("info string error: %v, position unchanged\n", err)
			break
		}
		u.pos, u.scratch = u.scratch, u.pos
	case "go":
		u.goSearch(parts[1:])
	case "stop":
		if u.current != nil {
			u.current.stop()
		}
	case "quit":
		return false
	case "d", "display":
		u.display()
	case "eval":
		u.out.printf("Evaluation: %+d (from %s's perspective, net %08X)\n", u.pos.evaluate(), [2]string{"White", "Black"}[u.pos.side], nnCRC)
	case "audit":
		u.audit()
	case "perft", "divide":
		u.perft(parts)
	case "help":
		u.out.printf("# %s - Available Commands:\n\n%s\n", EngineName, helpText)
	default:
		u.out.printf("info string unknown command %q, type help for the command list\n", parts[0])
	}
	return true
}

func (u *uci) setOption(parts []string) {
	name, value := parseSetOption(parts)
	switch {
	case strings.EqualFold(name, "Hash"):
		sizeMB, err := strconv.Atoi(value)
		if err != nil || sizeMB < 1 || sizeMB > 4096 {
			u.out.printf("info string error: Hash must be 1 to 4096 MB, got %q\n", value)
			return
		}
		u.stopSearch()
		initTT(sizeMB)
		u.out.printf("info string Hash set to %d MB\n", sizeMB)
	case strings.EqualFold(name, "Move Overhead"):
		ms, err := strconv.Atoi(value)
		if err != nil || ms < 0 || ms > 5000 {
			u.out.printf("info string error: Move Overhead must be 0 to 5000 ms, got %q\n", value)
			return
		}
		u.moveOverhead = int64(ms)
	default:
		u.out.printf("info string setoption %q = %q ignored\n", name, value)
	}
}

func parseSetOption(parts []string) (name, value string) {
	i := slices.Index(parts, "name")
	if i < 0 {
		return "", ""
	}
	rest := parts[i+1:]
	j := slices.Index(rest, "value")
	switch {
	case j < 0:
		return strings.Join(rest, " "), ""
	case j == 0:
		return "", ""
	}
	return strings.Join(rest[:j], " "), strings.Join(rest[j+1:], " ")
}

// parseGo reads the arguments of a UCI go command. Unknown tokens such as ponder or
// searchmoves are skipped. A bad or missing value is returned as an error but the
// rest is still used: the GUI waits for a bestmove either way.
func parseGo(args []string) (*TimeControl, error) {
	tc := &TimeControl{}
	var bad []string
	for i := 0; i < len(args); i++ {
		name := args[i]
		if name == "infinite" {
			tc.infinite = true
			continue
		}
		if !slices.Contains([]string{"wtime", "btime", "winc", "binc", "movestogo", "depth", "movetime"}, name) {
			continue
		}
		if i+1 == len(args) {
			bad = append(bad, name+" has no value")
			break
		}
		i++
		v, err := strconv.ParseInt(args[i], 10, 64)
		if err != nil {
			bad = append(bad, fmt.Sprintf("%s value %q is not a number", name, args[i]))
			continue
		}
		switch name {
		case "wtime":
			tc.wtime = v
		case "btime":
			tc.btime = v
		case "winc":
			tc.winc = v
		case "binc":
			tc.binc = v
		case "movestogo":
			tc.movestogo = int(v)
		case "depth":
			tc.depth = int(v)
		case "movetime":
			tc.movetime = v
		}
	}
	if len(bad) > 0 {
		return tc, fmt.Errorf("%s", strings.Join(bad, "; "))
	}
	return tc, nil
}

// goSearch starts searching the current position in the background.
func (u *uci) goSearch(args []string) {
	u.stopSearch()
	tc, err := parseGo(args)
	if err != nil {
		u.out.printf("info string error in go: %v\n", err)
	}
	tc.allocateTime(u.pos.side, u.moveOverhead)
	u.searcher.pos = *u.pos
	u.current = tc
	u.searchWG.Add(1)
	go u.runSearch(tc)
}

// runSearch searches and reports the best move. After go infinite the protocol
// allows the report only after stop, even when the search has reached MaxDepth.
func (u *uci) runSearch(tc *TimeControl) {
	defer u.searchWG.Done()
	move := u.searcher.search(tc)
	for tc.infinite && atomic.LoadInt32(&tc.stopped) == 0 {
		time.Sleep(time.Millisecond)
	}
	u.out.printf("bestmove %v\n", move)
}

// stopSearch halts the latest search and waits until it has printed its bestmove.
func (u *uci) stopSearch() {
	if u.current != nil {
		u.current.stop()
		u.searchWG.Wait()
		u.current = nil
	}
}

func (u *uci) display() {
	var b strings.Builder
	b.WriteString("\n   a b c d e f g h\n  ----------------\n")
	for r := 7; r >= 0; r-- {
		fmt.Fprintf(&b, "%d|", r+1)
		for f := 0; f < 8; f++ {
			if v := u.pos.square[r*8+f]; v >= 0 {
				fmt.Fprintf(&b, " %c", "PNBRQKpnbrqk"[(v>>3)*6+v&7])
			} else {
				b.WriteString(" .")
			}
		}
		fmt.Fprintf(&b, " |%d\n", r+1)
	}
	b.WriteString("  ----------------\n   a b c d e f g h\n")
	fmt.Fprintf(&b, "Side to move: %s\nHash: %x\n\n", [2]string{"White", "Black"}[u.pos.side], u.pos.hash)
	u.out.printf("%s", b.String())
}

// audit checks the incrementally updated state of the current position against a
// rebuild from its squares: occupancy, Zobrist hash and NNUE accumulator.
func (u *uci) audit() {
	p := u.pos
	report := func(ok bool, what, desync string) {
		if ok {
			u.out.printf("  - %s: OK\n", what)
		} else {
			u.out.printf("!! %s\n", desync)
		}
	}
	u.out.printf("# Starting internal state audit...\n")

	occ := Bitboard(0)
	for c := White; c <= Black; c++ {
		for pt := Pawn; pt <= King; pt++ {
			occ |= p.pieces[c][pt]
		}
	}
	report(occ == p.all, "Bitboard occupancy", fmt.Sprintf("BITBOARD DESYNC: pos.all (%x) != calculated (%x)", p.all, occ))

	hash := uint64(0)
	if p.side == Black {
		hash ^= zobristSide
	}
	hash ^= zobristCastleDiff[p.castle]
	if p.epSquare != -1 {
		hash ^= zobristEP[p.epSquare%8]
	}
	for sq, v := range p.square {
		if v >= 0 {
			hash ^= zobristPiece[v>>3][v&7][sq]
		}
	}
	report(hash == p.hash, "Zobrist hash", fmt.Sprintf("HASH DESYNC: pos.hash (%x) != calculated (%x)", p.hash, hash))

	p.nnBuild()
	var fresh [2][NNHidden]int16
	p.nnRefresh(&fresh, White)
	p.nnRefresh(&fresh, Black)
	report(fresh == p.acc[p.historyPly], "NNUE accumulator", "NNUE ACCUMULATOR DESYNC")
	u.out.printf("# Audit complete.\n")
}

// perft runs "perft <depth>" (node count and speed for each depth up to depth) or
// "divide <depth>" (node count per root move).
func (u *uci) perft(parts []string) {
	depth := 0
	if len(parts) == 2 {
		depth, _ = strconv.Atoi(parts[1])
	}
	if depth < 1 {
		u.out.printf("info string error: usage: %s <depth>, depth at least 1\n", parts[0])
		return
	}
	if parts[0] == "divide" {
		u.pos.perftDivide(depth, u.out)
		return
	}
	u.out.printf("\nRunning perft test...\nDepth    Nodes           Time        NPS\n---------------------------------------------\n")
	for d := 1; d <= depth; d++ {
		start := time.Now()
		count := u.pos.perft(d)
		elapsed := time.Since(start)
		nps := int64(0)
		if elapsed.Seconds() > 0 {
			nps = int64(float64(count) / elapsed.Seconds())
		}
		timeStr := fmt.Sprintf("%d ms", elapsed.Milliseconds())
		if elapsed >= time.Second {
			timeStr = fmt.Sprintf("%.2f s", elapsed.Seconds())
		}
		u.out.printf("%-8d %-15d %-11s %d\n", d, count, timeStr, nps)
	}
	u.out.printf("\n")
}

const helpText = `UCI Protocol Commands:
  uci                              - Initialize UCI mode
  isready                          - Check if engine is ready
  setoption name <id> value <x>    - Hash (MB) or Move Overhead (ms)
  ucinewgame                       - Start a new game
  position startpos                - Set starting position
  position startpos moves <moves>  - Set position after moves
  position fen <fen> [moves ...]   - Set position from a FEN
  go [options]                     - Start searching
      wtime <ms>                   - White's remaining time
      btime <ms>                   - Black's remaining time
      winc <ms>                    - White's increment per move
      binc <ms>                    - Black's increment per move
      movestogo <n>                - Moves until time control
      depth <n>                    - Search to fixed depth
      movetime <ms>                - Search for fixed time
      infinite                     - Search until stop
  stop                             - Stop searching
  quit                             - Exit engine

Additional Commands:
  d, display                       - Display current board
  eval                             - Show static evaluation
  perft <depth>                    - Run perft test
  divide <depth>                   - Run divide test
  audit                            - Audit state handling
  help                             - Show this help message

Example Usage:
  1. Start new game:
     ucinewgame

  2. Set position and make moves:
     position startpos moves e2e4 e7e5 g1f3

  3. Search with time control:
     go wtime 300000 btime 300000 winc 0 binc 0

  4. Search to depth 10:
     go depth 10

  5. Display current position:
     d`

func main() {
	if err := loadNet(netData); err != nil {
		fmt.Fprintln(os.Stderr, "soomi.nnue:", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, EngineName, "- UCI Chess Engine")
	fmt.Fprintln(os.Stderr, "Type 'help' for available commands or 'uci' to enter UCI mode")
	uciLoop(os.Stdin, os.Stdout)
}

// To make an executable
// set "GOEXPERIMENT=simd" && set GOAMD64=v3 && go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi.exe Soomi.go
