package main

import (
	"bufio"
	"fmt"
	"math"
	"math/bits"
	"os"
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
	MaxDepth                           = 50
	Infinity                           = 30000
	Mate                               = 29000
	AspirationBase                     = 50
	AspirationStartDepth               = 4
	DefaultMovesToGo                   = 20
	NodeCheckMaskSearch                = 1023
	DeltaMargin                        = 150
	LMRMinChildDepth                   = 3
	LMRLateMoveAfter                   = 3
	MateScoreGuard                     = 1000
	SEEPruneDepthMax                   = 8  // SEE prune only at depth <= this
	SEEQuietCoeff                      = 80 // quiet margin: -coeff * depth
	SEENoisyCoeff                      = 30 // capture margin: -coeff * depth * depth
	LMPDepthMax                        = 4  // LMP only at depth <= this
	LMPBase                            = 3  // quiet limit: LMPBase + depth * depth
	defaultTTSizeMB                    = 256
	scoreHash                          = 1000000
	scorePromoBase                     = 900000
	scoreCaptureBase                   = 800000
	scoreKiller1                       = 750000
	scoreKiller2                       = 740000
	scoreCountermove                   = 200000
	minTimeMs            int64         = 5
	continueMargin       time.Duration = 10 * time.Millisecond
	MaxGamePly                         = 1024
	ZobristSeed                        = 1070372
	totalPhase                         = 24
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
	ttFlagExact uint8 = 0
	ttFlagLower uint8 = 1
	ttFlagUpper uint8 = 2
)

const (
	PhaseScale   = 256
	MVVLVAWeight = 100
	MaxHistory   = 16384
	// pawnTableSize must be a power of 2 for bitwise AND mask to work.
	pawnTableSize = 131072 // 4MB table (131072 * 32 bytes)
)

// pawnEntry caches the evaluation of a specific pawn structure.
// Since pawns move very rarely compared to pieces, the pawn structure remains
// identical across dozens of consecutive search nodes.
// Same idea as tt, why evaluate if we know already?
type pawnEntry struct {
	key     uint64 // Zobrist hash of the pawn positions
	mgScore int    // Middle-game score of this pawn structure
	egScore int    // End-game score of this pawn structure
}

var (
	pawnTable         []pawnEntry
	pieceValues       = [6]int{89, 313, 317, 504, 1001, 20000}
	pst               [2][6][64]int
	pstEnd            [2][6][64]int
	piecePhase        = [6]int{0, 1, 1, 2, 4, 0}
	currentTC         atomic.Pointer[TimeControl]
	searchWG          sync.WaitGroup
	tt                *TranspositionTable
	rookMagics        [64]MagicEntry
	bishopMagics      [64]MagicEntry
	rookAttackTable   [102400]Bitboard
	bishopAttackTable [5248]Bitboard
	passedPawnBonus   = [8]int{0, 13, 14, 33, 55, 97, 154, 0}
	mobilityBonus     = [4][]int{
		{-14, 1, 9, 16, 23, 28, 31, 31, 19},                                                                              // Knight
		{-13, -4, 6, 14, 21, 26, 29, 35, 38, 40, 38, 39, 27, 23},                                                         // Bishop
		{-15, -2, 4, 11, 15, 21, 26, 30, 34, 38, 39, 42, 41, 37, 27},                                                     // Rook
		{9, 9, 13, 17, 18, 24, 29, 32, 37, 38, 41, 44, 45, 46, 48, 51, 52, 55, 51, 48, 47, 34, 30, 12, 1, -34, -24, -53}, // Queen
	}
	kingZoneMask       [64]Bitboard
	kingAttackerWeight = [6]int{0, 2, 1, 3, 3, 0} // P, N, B, R, Q, K (P and K usually 0 or special)
	history            [2][64][64]int
	countermoves       [2][64][64]Move
	lmrTable           [MaxDepth + 1][256]int
	lvaOrder           = [6]int{Pawn, Bishop, Knight, Rook, Queen, King}
	castleMask         [64]int
	pawnShieldMask     [2][64]Bitboard
	fileMasks          [8]Bitboard
	passedPawnMask     [2][64]Bitboard
)

const (
	isolatedPawnPenalty   = 14
	doubledPawnPenalty    = 9
	bonusBishopPair       = 36
	bonusRookOpenFile     = 18
	bonusRookSemiOpenFile = 15
	bonusPawnShield       = 15
	penaltyPawnStorm      = 2
	bonusKnightOutpost    = 31
	penaltyKingTropism    = 3
	bonusRookOn7th        = 17
)

/*
  ----------------------------------------------------------------------------------
   MAGIC BITBOARDS
  ----------------------------------------------------------------------------------
   Magic bitboards allows for instant sliding piece attacks

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
	material         [2]int
	psqScore         [2]int
	psqScoreEG       [2]int
	square           [64]int
	historyKeys      [MaxGamePly]uint64
	historyPly       int
	lastIrreversible int
	localNodes       int64
	pawnHash         uint64
	kingSq           [2]int
	phase            int
	seldepth         int
	searchStart      time.Time
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

/*
  ----------------------------------------------------------------------------------
   TAPERED EVALUATION (Phase Calculation)
  ----------------------------------------------------------------------------------
   Chess strategy changes a lot as pieces are exchanged. We use a "Phase" variable
   to interpolate between Middle Game (MG) and End Game (EG) scores.
   This combats evaluation discontinuity, a major source of instability

   Total Phase = 24 (Based on piece weights: N=1, B=1, R=2, Q=4)

   Score Formula:
      FinalScore = ( (MG_Score * Phase) + (EG_Score * (256 - Phase)) ) / 256

      Start Position (all 32 pieces)       King vs King
      (Phase = 256)                       (Phase = 0)
      [========================================]
      ^                                        ^
      |                                        |
   Use mostly MG values                     Use mostly EG values
*/

func (p *Position) computePhase() {
	p.phase = totalPhase
	for pt := Knight; pt <= Queen; pt++ {
		p.phase -= bits.OnesCount64(uint64(p.pieces[White][pt]|p.pieces[Black][pt])) * piecePhase[pt]
	}
}

func (p *Position) isEndgame() bool {
	return p.phase > 18
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

type ttEntry struct {
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
   2. Index = Hash % TableSize.
   3. Check if stored Key matches current Key.
   4. If Depth >= NeededDepth, use the stored Score immediately.
*/

type TranspositionTable struct {
	entries []ttEntry
	mask    uint64
	gen     uint32
}

// InitTT initializes the transposition table.
// Small note: it is recommended to only use power of 2 values
// As the engine will automatically round it down to the nearest power of 2 anyways.
// Example: You input setoption hash value 150. It will be rounded to 128.
// Recommended values: 64, 128, 256, 512, 1024...
func InitTT(sizeMB int) {
	// One ttEntry is two uint64s = 16 bytes. bits.Len64 finds how many bits the entry
	// count needs; shifting 1 by one less rounds it down to a power of 2 (100 -> 64).
	// size-1 is then a mask, so indexing can use AND instead of the much slower modulo.
	size := uint64(1) << (bits.Len64(uint64(sizeMB)*1024*1024/16) - 1)
	tt = &TranspositionTable{entries: make([]ttEntry, size), mask: size - 1}
}

func (t *TranspositionTable) Clear() {
	t.gen++
	if t.gen > 255 {
		t.gen = 1
		for i := range t.entries {
			t.entries[i] = ttEntry{}
		}
	}
}

// Probe decodes the packed word in place (layout above), which keeps it inlinable.
func (t *TranspositionTable) Probe(key uint64, minDepth int) (move Move, score int, flag uint8, found, usable bool) {
	e := t.entries[key&t.mask]
	if e.key != key {
		return
	}
	p := e.packed
	return Move(p >> 32), int(int16(p >> 16)), uint8(p & 3), true, uint8(p>>8) == uint8(t.gen) && int(p>>2&0x3F) >= minDepth
}

// Save stores a search result into memory.
// We use always replace, so we can get relevant entries
// Its simple but can overwrite good entries with trash ones
func (t *TranspositionTable) Save(key uint64, mv Move, score int, depth int, flag uint8) {
	packed := uint64(mv)<<32 | uint64(uint16(int16(score)))<<16 | uint64(uint8(t.gen))<<8 | uint64(uint8(min(depth, 63)))<<2 | uint64(flag&3)
	t.entries[key&t.mask] = ttEntry{key: key, packed: packed}
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

// Hashfull samples the first 1000 entries (the table always has at least 65536).
func (t *TranspositionTable) Hashfull() int {
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
	initPawnShield()
	initPassedPawnMask()
	initAttacks()
	initMagicBitboards()
	initEvaluation()
	initLMR()
	InitTT(defaultTTSizeMB)
	pawnTable = make([]pawnEntry, pawnTableSize)
}

func initPassedPawnMask() {
	for sq := 0; sq < 64; sq++ {
		r, f := sq/8, sq%8
		for nf := max(0, f-1); nf <= min(7, f+1); nf++ {
			for nr := r + 1; nr < 8; nr++ {
				passedPawnMask[White][sq] |= sqBB[nr*8+nf]
			}
			for nr := r - 1; nr >= 0; nr-- {
				passedPawnMask[Black][sq] |= sqBB[nr*8+nf]
			}
		}
	}
}

func initPawnShield() {
	for sq := 0; sq < 64; sq++ {
		r, f := sq/8, sq%8
		for nf := max(0, f-1); nf <= min(7, f+1); nf++ {
			if r <= 2 {
				pawnShieldMask[White][sq] |= sqBB[(r+1)*8+nf] | sqBB[(r+2)*8+nf]
			}
			if r >= 5 {
				pawnShieldMask[Black][sq] |= sqBB[(r-1)*8+nf] | sqBB[(r-2)*8+nf]
			}
		}
	}
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
			val := 0.5 + math.Log(float64(d))*math.Log(float64(m))/3.0
			lmrTable[d][m] = int(val)
		}
	}
}

func initEvaluation() {
	for i := 0; i < 8; i++ {
		fileMasks[i] = 0x0101010101010101 << i
	}
	for sq := 0; sq < 64; sq++ {
		// King ring plus one more rank each way; bits shifted off the board drop out
		mask := kingAttacks[sq] | sqBB[sq]
		kingZoneMask[sq] = mask | mask<<8 | mask>>8
	}
}

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

	pstEnd[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		17, 13, 19, 6, 9, 22, 9, 17,
		16, 13, 16, 12, 19, 22, 8, 13,
		27, 22, 16, 11, 11, 13, 21, 21,
		41, 35, 11, 15, 12, 14, 26, 28,
		62, 48, 39, 24, 8, 42, 39, 41,
		43, 55, 7, -21, -17, 45, 23, 27,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pstEnd[White][Knight] = [64]int{
		-24, -38, -15, -23, -25, -22, -21, -20,
		-58, -6, -15, -6, -6, -21, -20, -46,
		-25, 6, -4, 4, 6, -1, -12, -15,
		-7, -12, 14, 13, 11, 14, -4, -17,
		-16, 1, 6, 13, 9, 7, 1, -11,
		-14, -9, 7, 2, 3, -15, -6, -12,
		0, -1, -1, -12, -1, -11, -3, -20,
		-36, 0, -10, -18, -11, -14, -43, -87,
	}

	pstEnd[White][Bishop] = [64]int{
		16, -23, -14, -6, -8, -8, -14, -8,
		12, -14, -12, 1, -2, -2, -14, -10,
		-9, 1, 4, 4, 10, -2, -6, 6,
		-13, 7, 10, 3, -2, 11, -1, -12,
		-5, -7, 0, 6, 9, -15, 7, -7,
		-9, 2, 2, -3, 5, -4, -6, 10,
		10, -1, 10, -13, -2, 4, 14, 0,
		26, 2, 13, -2, 5, 6, 1, 32,
	}

	pstEnd[White][Rook] = [64]int{
		1, -10, -10, -6, -12, -6, -12, -11,
		2, -5, 2, -2, -4, -8, -13, 2,
		4, 1, 11, 6, 9, 4, 0, -5,
		16, 23, 28, 10, 8, 20, 11, 7,
		22, 20, 15, 14, 5, 17, 19, 12,
		27, 8, 16, 17, 8, 11, 6, 16,
		24, 23, 24, 22, 12, 16, 26, 26,
		22, 12, 9, 18, 13, 9, 2, 13,
	}

	pstEnd[White][Queen] = [64]int{
		-33, -82, -80, -64, -88, -78, -115, -63,
		-56, -53, -41, -60, -54, -83, -88, -83,
		-21, -15, -1, 1, -10, 0, -12, -30,
		-40, -6, 21, 34, 25, 18, 15, 3,
		-18, 25, 9, 56, 24, 49, 50, 17,
		-6, 0, 37, 20, 51, 18, 36, 27,
		16, 30, 24, 24, 20, 19, 7, 13,
		-9, -10, -19, 0, 12, 2, 9, -23,
	}

	pstEnd[White][King] = [64]int{
		-59, -37, -27, -24, -26, -32, -31, -42,
		-34, -17, -5, -7, -3, -6, -19, -23,
		-11, -2, 8, 16, 18, 9, 4, -12,
		-7, 18, 24, 29, 24, 26, 18, -3,
		0, 37, 30, 42, 38, 31, 28, 10,
		-3, 29, 37, 36, 28, 36, 54, 13,
		-10, 7, 19, 21, 17, 33, 18, -17,
		-79, -11, 21, -21, -8, -7, -3, -43,
	}

	// Mirror for black
	for pt := 0; pt < 6; pt++ {
		for sq := 0; sq < 64; sq++ {
			bsq := sq ^ 56
			pst[Black][pt][sq] = pst[White][pt][bsq]
			pstEnd[Black][pt][sq] = pstEnd[White][pt][bsq]
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

func makeMoves(from, to, flags int) Move {
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

func NewPosition() *Position {
	p := &Position{}
	p.setStartPos()
	return p
}

func (p *Position) setStartPos() {
	p.setFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
}

func (p *Position) setFEN(fen string) {
	*p = Position{}
	for i := range p.square {
		p.square[i] = -1
	}
	p.epSquare = -1
	parts := strings.Fields(fen)

	// 1. Da pieces
	for r, rank := range strings.Split(parts[0], "/") {
		file := 0
		for _, ch := range rank {
			if ch >= '1' && ch <= '8' {
				file += int(ch - '0')
				continue
			}
			sq := (7-r)*8 + file
			file++
			color := White
			if ch >= 'a' {
				color = Black
				ch -= 32
			}
			pt := strings.IndexByte("PNBRQK", byte(ch))
			if pt < 0 {
				continue
			}
			bb := sqBB[sq]
			p.pieces[color][pt] |= bb
			p.occupied[color] |= bb
			p.all |= bb
			p.square[sq] = (color << 3) | pt
			p.material[color] += pieceValues[pt]
			p.psqScore[color] += pst[color][pt][sq]
			p.psqScoreEG[color] += pstEnd[color][pt][sq]
			p.hash ^= zobristPiece[color][pt][sq]
			if pt == Pawn {
				p.pawnHash ^= zobristPiece[color][Pawn][sq]
			}
		}
	}

	// 2. Side to move
	if len(parts) >= 2 && parts[1] == "b" {
		p.side = Black
		p.hash ^= zobristSide
	}

	// 3. Castling
	if len(parts) >= 3 {
		for _, ch := range parts[2] {
			if i := strings.IndexRune("KQkq", ch); i >= 0 {
				p.castle |= 1 << i
			}
		}
		p.hash ^= zobristCastleDiff[p.castle]
	}

	// 4. The french pawn-move
	if len(parts) >= 4 && parts[3] != "-" {
		f, r := int(parts[3][0]-'a'), int(parts[3][1]-'1')
		p.epSquare = r*8 + f
		p.hash ^= zobristEP[f]
	}

	// 5. Clock
	if len(parts) >= 5 {
		p.halfmove, _ = strconv.Atoi(parts[4])
	}

	// Extras
	p.kingSq[White] = bits.TrailingZeros64(uint64(p.pieces[White][King]))
	p.kingSq[Black] = bits.TrailingZeros64(uint64(p.pieces[Black][King]))
	p.historyKeys[0] = p.hash
	p.historyPly = 0
	p.lastIrreversible = 0
	p.computePhase()
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
					buf[i] = makeMoves(from, to, flag)
					i++
				}
			}
			for att := pawnAttacks[us][from] & occThem; att != 0; {
				to := popLSB(&att)
				for flag := FlagPromoCQ; flag >= FlagPromoCN; flag-- {
					buf[i] = makeMoves(from, to, flag)
					i++
				}
			}
		} else {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				buf[i] = makeMoves(from, to, FlagQuiet)
				i++
				if from>>3 == dblRank && occAll&sqBB[from+2*push] == 0 {
					buf[i] = makeMoves(from, from+2*push, FlagQuiet)
					i++
				}
			}
			for att := pawnAttacks[us][from] & occThem; att != 0; {
				buf[i] = makeMoves(from, popLSB(&att), FlagCapture)
				i++
			}
			if ep >= 0 && pawnAttacks[us][from]&sqBB[ep] != 0 {
				buf[i] = makeMoves(from, ep, FlagEP)
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
				buf[i] = makeMoves(from, to, flag)
				i++
			}
		}
	}

	// castle bits: 1=K 2=Q 4=k 8=q; shift the side's pair down to bits 1 and 2
	if rights, base := p.castle>>(2*us), 56*us; !capturesOnly && rights&3 != 0 && !p.inCheck() {
		if rights&1 != 0 && occAll&(Bitboard(0x60)<<base) == 0 {
			buf[i] = makeMoves(base+4, base+6, FlagCastle)
			i++
		}
		if rights&2 != 0 && occAll&(Bitboard(0x0E)<<base) == 0 {
			buf[i] = makeMoves(base+4, base+2, FlagCastle)
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
		if capturedPiece == Pawn {
			p.pawnHash ^= zobristPiece[them][Pawn][capSq]
		}
		p.material[them] -= pieceValues[capturedPiece]
		p.phase += piecePhase[capturedPiece]
		p.psqScore[them] -= pst[them][capturedPiece][capSq]
		p.psqScoreEG[them] -= pstEnd[them][capturedPiece][capSq]
		p.square[capSq] = -1
	}

	if flags == FlagCastle {
		kingBB := sqBB[from] | sqBB[to]
		p.pieces[us][King] ^= kingBB
		p.occupied[us] ^= kingBB
		p.all ^= kingBB
		h ^= zobristPiece[us][King][from] ^ zobristPiece[us][King][to]
		p.psqScore[us] += pst[us][King][to] - pst[us][King][from]
		p.psqScoreEG[us] += pstEnd[us][King][to] - pstEnd[us][King][from]
		p.square[from] = -1
		p.square[to] = (us << 3) | King
		p.kingSq[us] = to
		rf, rt := from+3, from+1 // king side
		if to < from {
			rf, rt = from-4, from-1
		}
		rookBB := sqBB[rf] | sqBB[rt]
		p.pieces[us][Rook] ^= rookBB
		p.occupied[us] ^= rookBB
		p.all ^= rookBB
		h ^= zobristPiece[us][Rook][rf] ^ zobristPiece[us][Rook][rt]
		p.psqScore[us] += pst[us][Rook][rt] - pst[us][Rook][rf]
		p.psqScoreEG[us] += pstEnd[us][Rook][rt] - pstEnd[us][Rook][rf]
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		promoType := (flags & 3) + Knight
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][Pawn] ^= sqBB[from]
		p.pieces[us][promoType] ^= sqBB[to]
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		h ^= zobristPiece[us][Pawn][from]
		p.pawnHash ^= zobristPiece[us][Pawn][from]
		p.material[us] -= pieceValues[Pawn]
		p.psqScore[us] -= pst[us][Pawn][from]
		p.psqScoreEG[us] -= pstEnd[us][Pawn][from]
		p.square[from] = -1

		h ^= zobristPiece[us][promoType][to]

		p.material[us] += pieceValues[promoType]
		p.phase -= piecePhase[promoType]
		p.psqScore[us] += pst[us][promoType][to]
		p.psqScoreEG[us] += pstEnd[us][promoType][to]
		p.square[to] = (us << 3) | promoType

	} else {
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][movingPiece] ^= moveBB
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		h ^= zobristPiece[us][movingPiece][from] ^ zobristPiece[us][movingPiece][to]
		if movingPiece == Pawn {
			p.pawnHash ^= zobristPiece[us][Pawn][from] ^ zobristPiece[us][Pawn][to]
		}
		p.psqScore[us] += pst[us][movingPiece][to] - pst[us][movingPiece][from]
		p.psqScoreEG[us] += pstEnd[us][movingPiece][to] - pstEnd[us][movingPiece][from]
		p.square[from] = -1
		p.square[to] = (us << 3) | movingPiece
		if movingPiece == King {
			p.kingSq[us] = to
		}

		if movingPiece == Pawn && abs(to-from) == 16 {
			p.epSquare = (from + to) / 2
			h ^= zobristEP[p.epSquare%8]
		}
	}

	p.castle &= castleMask[from] & castleMask[to]

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
		p.psqScore[us] += pst[us][King][from] - pst[us][King][to]
		p.psqScoreEG[us] += pstEnd[us][King][from] - pstEnd[us][King][to]
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
		p.psqScore[us] += pst[us][Rook][rt] - pst[us][Rook][rf]
		p.psqScoreEG[us] += pstEnd[us][Rook][rt] - pstEnd[us][Rook][rf]
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		promoType := (flags & 3) + Knight
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][promoType] ^= sqBB[to]
		p.pieces[us][Pawn] ^= sqBB[from]
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		p.material[us] -= pieceValues[promoType]
		p.psqScore[us] -= pst[us][promoType][to]
		p.psqScoreEG[us] -= pstEnd[us][promoType][to]
		p.phase += piecePhase[promoType]
		p.square[to] = -1
		p.material[us] += pieceValues[Pawn]
		p.psqScore[us] += pst[us][Pawn][from]
		p.psqScoreEG[us] += pstEnd[us][Pawn][from]
		p.square[from] = (us << 3) | Pawn
		p.pawnHash ^= zobristPiece[us][Pawn][from]

	} else {
		movingPt := p.square[to] & 7
		moveBB := sqBB[from] | sqBB[to]
		p.pieces[us][movingPt] ^= moveBB
		p.occupied[us] ^= moveBB
		p.all ^= moveBB
		p.psqScore[us] += pst[us][movingPt][from] - pst[us][movingPt][to]
		p.psqScoreEG[us] += pstEnd[us][movingPt][from] - pstEnd[us][movingPt][to]
		p.square[to] = -1
		p.square[from] = (us << 3) | movingPt
		// Update da kingsq
		if movingPt == King {
			p.kingSq[us] = from
		}
		if movingPt == Pawn {
			p.pawnHash ^= zobristPiece[us][Pawn][from] ^ zobristPiece[us][Pawn][to]
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
		p.material[them] += pieceValues[capturedPiece]
		p.psqScore[them] += pst[them][capturedPiece][capSq]
		p.psqScoreEG[them] += pstEnd[them][capturedPiece][capSq]
		p.square[capSq] = (them << 3) | capturedPiece
		p.phase -= piecePhase[capturedPiece]
		if capturedPiece == Pawn {
			p.pawnHash ^= zobristPiece[them][Pawn][capSq]
		}
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

func (p *Position) evalBishopPair() int {
	score := 0
	pawnCount := bits.OnesCount64(uint64(p.pieces[White][Pawn] | p.pieces[Black][Pawn]))
	// Bishop pair is worth more in open positions (less pawns)
	dynamicBonus := bonusBishopPair + (16-pawnCount)*2

	if bits.OnesCount64(uint64(p.pieces[White][Bishop])) >= 2 {
		score += dynamicBonus
	}
	if bits.OnesCount64(uint64(p.pieces[Black][Bishop])) >= 2 {
		score -= dynamicBonus
	}
	return score
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

func updateHistory(side, from, to, bonus int) {
	bonus = max(-400, min(400, bonus))
	history[side][from][to] += bonus - history[side][from][to]*abs(bonus)/MaxHistory
}

func (p *Position) evalPawns() (mg, eg int) {
	// Explained this during initTT, same logic here
	idx := p.pawnHash & (pawnTableSize - 1)
	entry := &pawnTable[idx]

	// Check if we have evaluated this same pawn struct before
	if entry.key == p.pawnHash {
		// We have evaluated this before, return cached score
		return entry.mgScore, entry.egScore
	}

	for side := White; side <= Black; side++ {
		own, enemy, sign := p.pieces[side][Pawn], p.pieces[side^1][Pawn], 1-2*side
		for bb := own; bb != 0; {
			sq := popLSB(&bb)
			file := sq % 8
			penalty := 0
			if own&fileMasks[file]&^sqBB[sq] != 0 { // doubled
				penalty += doubledPawnPenalty
			}
			if (file == 0 || own&fileMasks[file-1] == 0) && (file == 7 || own&fileMasks[file+1] == 0) { // isolated
				penalty += isolatedPawnPenalty
			}
			mg -= sign * penalty
			eg -= sign * penalty
			if enemy&passedPawnMask[side][sq] == 0 { // passed
				bonus := passedPawnBonus[(sq/8)^(7*side)]
				mg += sign * (bonus / 2)
				eg += sign * bonus
			}
		}
	}

	// Save the newly calculated scores into the pawn hash table.
	// This uses always replace like tt
	// because its simple, reliable mostly
	// And these dont change that often usually
	entry.key = p.pawnHash
	entry.mgScore = mg
	entry.egScore = eg

	return mg, eg
}

func (p *Position) evalMobility() (mg, eg int) {
	// Squares attacked by enemy pawns do not count as mobility
	danger := [2]Bitboard{
		((p.pieces[Black][Pawn] & ^Bitboard(0x8080808080808080)) >> 7) | ((p.pieces[Black][Pawn] & ^Bitboard(0x0101010101010101)) >> 9),
		((p.pieces[White][Pawn] & ^Bitboard(0x0101010101010101)) << 7) | ((p.pieces[White][Pawn] & ^Bitboard(0x8080808080808080)) << 9),
	}
	for side := White; side <= Black; side++ {
		safe := ^p.occupied[side] & ^danger[side]
		score := 0
		for bb := p.pieces[side][Knight]; bb != 0; {
			score += mobilityBonus[0][bits.OnesCount64(uint64(knightAttacks[popLSB(&bb)]&safe))]
		}
		for bb := p.pieces[side][Bishop]; bb != 0; {
			score += mobilityBonus[1][bits.OnesCount64(uint64(bishopAttacks(popLSB(&bb), p.all)&safe))]
		}
		for bb := p.pieces[side][Rook]; bb != 0; {
			score += mobilityBonus[2][bits.OnesCount64(uint64(rookAttacks(popLSB(&bb), p.all)&safe))]
		}
		for bb := p.pieces[side][Queen]; bb != 0; {
			sq := popLSB(&bb)
			score += mobilityBonus[3][bits.OnesCount64(uint64((bishopAttacks(sq, p.all)|rookAttacks(sq, p.all))&safe))]
		}
		mg += score * (1 - 2*side)
	}
	return mg, mg // mobilityBonus has no separate endgame table
}

func (p *Position) evalKingSafety() int {
	mg := 0
	for side := White; side <= Black; side++ {
		zone := kingZoneMask[p.kingSq[side]]
		attackers, attackUnits := 0, 0
		for pt := Knight; pt <= Queen; pt++ {
			for bb := p.pieces[side^1][pt]; bb != 0; {
				sq := popLSB(&bb)
				var attacks Bitboard
				switch pt {
				case Knight:
					attacks = knightAttacks[sq]
				case Bishop:
					attacks = bishopAttacks(sq, p.all)
				case Rook:
					attacks = rookAttacks(sq, p.all)
				case Queen:
					attacks = bishopAttacks(sq, p.all) | rookAttacks(sq, p.all)
				}
				if attacks&zone != 0 {
					attackers++
					attackUnits += kingAttackerWeight[pt]
				}
			}
		}
		score := bits.OnesCount64(uint64(p.pieces[side][Pawn]&pawnShieldMask[side][p.kingSq[side]])) * bonusPawnShield
		if attackers > 1 {
			score -= attackUnits * attackUnits
		}
		mg += score * (1 - 2*side)
	}
	return mg
}

func (p *Position) evalPawnStorm() int {
	score := 0
	for side := White; side <= Black; side++ {
		kingSq, sign := p.kingSq[side], 1-2*side
		for bb := p.pieces[side^1][Pawn] & passedPawnMask[side][kingSq]; bb != 0; {
			dist := (popLSB(&bb)/8 - kingSq/8) * sign
			score -= sign * penaltyPawnStorm * (6 - dist)
		}
	}
	return score
}

func (p *Position) evalOutposts() (mg, eg int) {
	// Knight and bishop outposts: relative rank 4-6, pawn-supported, and no enemy pawn can challenge
	for side := White; side <= Black; side++ {
		sign := 1 - 2*side
		for pt := Knight; pt <= Bishop; pt++ {
			bonus := bonusKnightOutpost
			if pt == Bishop {
				bonus /= 2
			}
			for bb := p.pieces[side][pt]; bb != 0; {
				sq := popLSB(&bb)
				if rel := (sq / 8) ^ (7 * side); rel >= 3 && rel <= 5 &&
					pawnAttacks[side^1][sq]&p.pieces[side][Pawn] != 0 &&
					p.pieces[side^1][Pawn]&passedPawnMask[side][sq]&^fileMasks[sq%8] == 0 {
					mg += sign * bonus
					eg += sign * (bonus / 2)
				}
			}
		}
	}
	return mg, eg
}

func (p *Position) evalTropism() int {
	score := 0
	for side := White; side <= Black; side++ {
		kr, kf := p.kingSq[side]/8, p.kingSq[side]%8
		closeness := 0
		for bb := p.occupied[side^1] &^ (p.pieces[side^1][Pawn] | p.pieces[side^1][King]); bb != 0; {
			sq := popLSB(&bb)
			closeness += 14 - (abs(sq/8-kr) + abs(sq%8-kf))
		}
		score -= (1 - 2*side) * penaltyKingTropism * closeness
	}
	return score
}

func (p *Position) evalRooksOnFiles() int {
	score := 0
	for side := White; side <= Black; side++ {
		s := 0
		enemyKingOnBackRank := (p.kingSq[side^1]/8)^(7*side) == 7
		for bb := p.pieces[side][Rook]; bb != 0; {
			sq := popLSB(&bb)
			if file := fileMasks[sq%8]; p.pieces[side][Pawn]&file == 0 {
				if p.pieces[side^1][Pawn]&file == 0 {
					s += bonusRookOpenFile
				} else {
					s += bonusRookSemiOpenFile
				}
			}
			if (sq/8)^(7*side) == 6 && enemyKingOnBackRank {
				s += bonusRookOn7th
			}
		}
		score += (1 - 2*side) * s
	}
	return score
}

func (p *Position) evaluate() int {
	pawnMG, pawnEG := p.evalPawns()
	mobilityMG, mobilityEG := p.evalMobility()
	outpostMG, outpostEG := p.evalOutposts()
	// Terms scored the same in both phases
	both := p.material[White] - p.material[Black] + p.evalBishopPair() + p.evalRooksOnFiles()
	mgScore := both + p.psqScore[White] - p.psqScore[Black] + pawnMG + mobilityMG + outpostMG +
		p.evalKingSafety() + p.evalTropism() + p.evalPawnStorm() +
		20 - 40*p.side // tempo: +20 with White to move, -20 with Black
	egScore := both + p.psqScoreEG[White] - p.psqScoreEG[Black] + pawnEG + mobilityEG + outpostEG
	phaseScaled := ((totalPhase-p.phase)*PhaseScale + totalPhase/2) / totalPhase
	score := egScore + ((mgScore-egScore)*phaseScaled)/PhaseScale
	return score * (1 - 2*p.side) // side-to-move perspective
}

/*
  ----------------------------------------------------------------------------------
   MOVE ORDERING
  ----------------------------------------------------------------------------------
   To prune the search tree effectively, we must search the best moves first.

   Sorting Priority List:
   1. Hash Move      --> The best move found at the previous depth (da best)
   2. Promotions     --> Queen promotions are tasty
   3. Good Captures  --> Positive SEE captures sorted via MVV-LVA
   4. Killer Moves   --> Quiet moves that caused a catoff in a sibling node
   5. Countermoves   --> Moves that historically responded well to the previous move (arguments?)
   6. History/PST    --> Moves that have been good in this game or improve PST position
*/

func clearHeuristics() {
	history = [2][64][64]int{}
	countermoves = [2][64][64]Move{}
}

func (p *Position) orderMoves(moves []Move, bestMove Move, killer1, killer2 Move, prevMove Move) []Move {
	var stackScores [256]int
	scores := stackScores[:len(moves)]
	for i, m := range moves {
		switch {
		case m == bestMove:
			scores[i] = scoreHash
		case m.isPromo() || m.isCapture():
			scores[i] = p.scoreNoisy(m)
		case m == killer1:
			scores[i] = scoreKiller1
		case m == killer2:
			scores[i] = scoreKiller2
		default:
			from, to := m.from(), m.to()
			pt := p.square[from] & 7
			// History, plus a PST delta for moves without history
			scores[i] = history[p.side][from][to] + pst[p.side][pt][to] - pst[p.side][pt][from]
			if prevMove != 0 && m == countermoves[p.side][prevMove.from()][prevMove.to()] {
				scores[i] += scoreCountermove
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
		return scorePromoBase + pieceValues[m.promoType()]
	}
	seeVal := p.see(m)
	if seeVal < 0 {
		return seeVal
	}
	victim := Pawn
	if vt := p.square[m.to()]; vt != -1 {
		victim = vt & 7
	}
	return scoreCaptureBase + seeVal + pieceValues[victim]*MVVLVAWeight - pieceValues[p.square[m.from()]&7]
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

func (p *Position) quiesce(alpha, beta, ply int, tc *TimeControl) int {
	p.seldepth = max(p.seldepth, ply)
	if ply >= MaxDepth {
		return p.evaluate()
	}
	p.localNodes++
	if p.localNodes&NodeCheckMaskSearch == 0 && tc.shouldStop() {
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
		if tc.shouldStop() {
			return alpha
		}
		// Prune bad caps, not in check so we dont prune check responses
		if !inCheck && scores[i] < 0 {
			continue
		}
		if !p.isLegal(m) {
			continue
		}
		legalCount++

		undo := p.makeMove(m)
		score := -p.quiesce(-beta, -alpha, ply+1, tc)
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

   Key Optimizations used here:
   1. Transposition Table (TT): Cache results to avoid re-searching identical positions.
   2. Null Move Pruning (NMP): "Pass" the move; if still safe, the position is too good (cutoff).
   3. Late Move Reductions (LMR): Search moves that were ordered as less promising at lower depth first.
   4. Quiescence Search: At leaf nodes, play out captures to avoid "horizon effects".

   Alpha (α): Best score the maximizing player can guarantee so far.
   Beta  (β): Best score the minimizing player can guarantee so far.
   Condition: If Score >= Beta, we have a "Cutoff" (branch is too good, opponent won't allow it).
*/

func (p *Position) negamax(depth, alpha, beta, ply int, pvNode bool, tc *TimeControl, ss *[MaxDepth + 1]SearchStack, prevMove Move) int {
	p.seldepth = max(p.seldepth, ply)
	if ply >= MaxDepth {
		return p.evaluate()
	}
	p.localNodes++
	if p.localNodes&NodeCheckMaskSearch == 0 && tc.shouldStop() { // time check
		return alpha
	}

	inCheck := p.inCheck()

	if inCheck {
		depth++
	}

	if depth <= 0 {
		return p.quiesce(alpha, beta, ply, tc)
	}

	// TT lookup
	origAlpha := alpha
	var hashMove Move
	if move, score, flag, found, usable := tt.Probe(p.hash, depth); found {
		hashMove = move
		if !pvNode {
			// Mate scores are stored relative to this node and are usable at any depth
			if score > Mate-MateScoreGuard {
				score, usable = score-ply, true
			} else if score < -Mate+MateScoreGuard {
				score, usable = score+ply, true
			}
			if usable && (flag == ttFlagExact || (flag == ttFlagLower && score >= beta) || (flag == ttFlagUpper && score <= alpha)) {
				return score
			}
		}
	}

	// IIR
	if depth >= 4 && hashMove == 0 {
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

	// RFP check
	if depth <= 8 && !inCheck && !p.isEndgame() {
		eval := p.evaluate()
		// If we are far above beta, we can return soft fail
		if eval >= beta+DeltaMargin*depth {
			return eval
		}
	}

	// Adaptive Null Move Pruning
	if depth >= 3 && !inCheck && !p.isEndgame() {
		R := 3 + depth/6

		undo := p.makeNullMove()
		score := -p.negamax(depth-1-R, -beta, -beta+1, ply+1, false, tc, ss, 0)
		p.unmakeNullMove(undo)

		if score >= beta {
			return beta
		}
	}

	// ProbCut
	if depth >= 5 && !inCheck && !p.isEndgame() {
		probBeta := beta + 200
		if probBeta <= Mate-MateScoreGuard {
			score := p.negamax(depth-4, probBeta-1, probBeta, ply+1, false, tc, ss, prevMove)
			if score >= probBeta {
				// Soft fail
				return score
			}
		}
	}

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	moves := p.orderMoves(movesArr[:n], hashMove, ss[ply].killer1, ss[ply].killer2, prevMove)

	bestMove := Move(0)
	bestScore := -Infinity
	legalMoves := 0
	var quietsTried [256]Move
	quietCount := 0

	for _, m := range moves {
		if tc.shouldStop() {
			return alpha
		}
		if !p.isLegal(m) {
			continue
		}
		legalMoves++
		if ply == 0 && depth > 4 && time.Since(p.searchStart) >= 500*time.Millisecond {
			fmt.Printf("info depth %d currmove %v currmovenumber %d\n", depth, m, legalMoves)
		}
		isQuiet := !m.isCapture() && !m.isPromo()

		// LMP: at shallow depth, stop searching quiets once enough were tried
		if depth <= LMPDepthMax && isQuiet && !inCheck &&
			m != hashMove && bestScore > -Mate+MaxDepth &&
			quietCount >= LMPBase+depth*depth {
			continue
		}

		// SEE pruning: skip shallow moves that lose material
		if depth <= SEEPruneDepthMax && legalMoves > 1 && !inCheck &&
			m != hashMove && !m.isPromo() && bestScore > -Mate+MaxDepth {
			seeMargin := -SEENoisyCoeff * depth * depth
			if isQuiet {
				seeMargin = -SEEQuietCoeff * depth
			}
			if p.see(m) < seeMargin {
				continue
			}
		}

		if isQuiet {
			quietsTried[quietCount] = m
			quietCount++
		}
		undo := p.makeMove(m)
		ss[ply+1].pvLen = 0
		var score int

		if p.halfmove >= 100 || p.isRepetition() || p.isInsufficientMaterial() {
			// A draw is worth exactly 0
			score = 0
		} else {
			// Late move reductions & Principal variation search
			// Late or reducible moves get a zero-window probe first, then a
			// full-window re-search only if they beat alpha
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
					if history[p.side^1][m.from()][m.to()] < 0 {
						red++
					}
					d = max(1, childDepth-red)
				}
				score = -p.negamax(d, -alpha-1, -alpha, ply+1, false, tc, ss, m)
			}
			if !zeroWindow || score > alpha {
				score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvNode, tc, ss, m)
			}
		}

		p.unmakeMove(m, undo)

		if score >= beta {
			// Update killers
			if isQuiet && m != hashMove {
				k := &ss[ply]
				if m != k.killer1 {
					k.killer2, k.killer1 = k.killer1, m
				}
				// History Bonus
				bonus := depth * depth
				updateHistory(p.side, m.from(), m.to(), bonus)
				// History Malus
				for i := 0; i < quietCount-1; i++ {
					updateHistory(p.side, quietsTried[i].from(), quietsTried[i].to(), -bonus)
				}
				// Countermove
				countermoves[p.side][prevMove.from()][prevMove.to()] = m
			}

			// Store in transposition table
			tt.Save(p.hash, m, scoreToTT(score, ply), depth, ttFlagLower)
			return score
		}

		// Update best move
		if score > bestScore {
			bestScore = score
			bestMove = m
		}

		// Update alpha
		if score > alpha {
			alpha = score
			if pvNode {
				n := copy(ss[ply].pv[1:], ss[ply+1].pv[:ss[ply+1].pvLen])
				ss[ply].pv[0], ss[ply].pvLen = m, n+1
			}
		}
	}

	// Handle draw & checkmate results
	if legalMoves == 0 {
		score := 0
		if inCheck {
			score = -Mate + ply
		}
		tt.Save(p.hash, 0, scoreToTT(score, ply), depth, ttFlagExact)
		return score
	}

	flag := ttFlagExact
	if bestScore <= origAlpha {
		flag = ttFlagUpper
	}
	tt.Save(p.hash, bestMove, scoreToTT(bestScore, ply), depth, flag)
	return bestScore
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
	[Time Up!] -> Return result, even if iteration is unfinished. This is safe because even though its an unfinished iteration,
	We searched the previous completed iteration fully, sorted that move as likely the best for this iteration as well,
	so we either return that or an even better move.

   Benefits:
   1. Time Management: We always have a "best move so far" if we must stop abruptly (as we do in chess).
   2. Move Ordering: The BestMove from Depth X-1 is the first move searched at Depth X.
*/

// printInfo emits one UCI info line. bound is "", "lowerbound" or "upperbound".
func (p *Position) printInfo(depth, score int, pv []Move, elapsed time.Duration, bound string) {
	nps := int64(0)
	if elapsed > 0 {
		nps = int64(float64(p.localNodes) / elapsed.Seconds())
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
	fmt.Printf("info depth %d seldepth %d score %s nodes %d time %d nps %d hashfull %d",
		depth, p.seldepth, scoreStr, p.localNodes, elapsed.Milliseconds(), nps, tt.Hashfull())
	if len(pv) > 0 {
		fmt.Print(" pv")
		for _, m := range pv {
			fmt.Printf(" %v", m)
		}
	}
	fmt.Println()
}

func (p *Position) search(tc *TimeControl) Move {
	var bestMove Move
	var ss [MaxDepth + 1]SearchStack

	maxDepth := tc.depth
	if maxDepth == 0 || tc.infinite {
		maxDepth = MaxDepth
	}

	// Start timers
	p.localNodes = 0
	p.seldepth = 0
	start := time.Now()
	p.searchStart = start
	var prevScore int
	var prevBestMove Move
	stableIterations := 0
	lastIterElapsed := time.Duration(0)
	for depth := 1; depth <= maxDepth; depth++ {
		p.seldepth = depth
		ss[0].pvLen = 0
		var score int

		// Aspiration windows
		if depth >= AspirationStartDepth {
			window := AspirationBase
			low, high := prevScore-window, prevScore+window
			for {
				low, high = max(low, -Infinity), min(high, Infinity)
				score = p.negamax(depth, low, high, 0, true, tc, &ss, 0)

				if tc.shouldStop() {
					break
				}

				if score <= low {
					// Failed low: true score is at most this
					p.printInfo(depth, score, ss[0].pv[:ss[0].pvLen], time.Since(start), "upperbound")
					low -= window
					window *= 2
				} else if score >= high {
					// Failed high: true score is at least this
					p.printInfo(depth, score, ss[0].pv[:ss[0].pvLen], time.Since(start), "lowerbound")
					high += window
					window *= 2
				} else {
					break
				}

				if window >= 1000 {
					score = p.negamax(depth, -Infinity, Infinity, 0, true, tc, &ss, 0)
					break
				}
			}
		} else {
			score = p.negamax(depth, -Infinity, Infinity, 0, true, tc, &ss, 0)
		}
		elapsed := time.Since(start)

		if tc.shouldStop() {
			break
		}

		pv := ss[0].pv[:ss[0].pvLen]
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
		p.printInfo(depth, score, pv, elapsed, "")

		// Stability and score drop heuristics
		scale := 1.0
		if depth >= 5 {
			if stableIterations >= 3 {
				// Best move has remained stable for 3+ depths: save time!
				scale *= 0.75
			} else if stableIterations == 0 {
				// Best move changed this iteration: invest more time!
				scale *= 1.25
			}

			if prevScore != 0 && score < prevScore-40 {
				// Unexpected score drop: think harder to find defense
				scale *= 1.20
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

// -------------------------------------------
// TIME CONTROL
// -------------------------------------------
func (tc *TimeControl) Stop() {
	atomic.StoreInt32(&tc.stopped, 1)
}

/*
  ----------------------------------------------------------------------------------
   TIME MANAGEMENT HEURISTICS
  ----------------------------------------------------------------------------------
   Deciding how much time to spend on a move is hard balance to find.
   - Too little: We play hasty, weak moves.
   - Too much: We likely flag (run out of time) later in the game.

   Strategy:
   1. Moves To Go: Assume the game lasts ~20-30-40 more moves (Can be whatever you like).
   2. Increment: Always safely bank on the increment (winc/binc).
   3. Safety Buffer: Subtract a margin (minTimeMs) to account for possible GUI lag.

   Formula:
    TimeForMove = (RemainingTime / MovesToGo) + Increment
*/

func (tc *TimeControl) allocateTime(side int) {
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

	avail := max(t-minTimeMs, 0)
	// Soft target: fair fraction of remaining time plus 3/4 of increment.
	// With nothing available both limits bottom out at minTimeMs, so no special case.
	optimum := max(min(avail/mtg+(i*3)/4, avail), minTimeMs)
	// Hard emergency ceiling: up to 3.5x soft target, strictly capped at available time
	maxTime := max(min(optimum*7/2, avail), minTimeMs)

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
	if remainHard <= continueMargin {
		return false
	}

	// 3. Project next iteration time (~2x the completed iteration)
	if next := iterTime * 2; iterTime > 0 && (elapsed+next > softTarget*3/2 || next+continueMargin > remainHard) {
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

func (p *Position) perftDivide(depth int) {
	var moves [256]Move
	total := 0
	for _, m := range moves[:p.generateMovesTo(moves[:], false)] {
		if p.isLegal(m) {
			undo := p.makeMove(m)
			count := p.perft(depth - 1)
			p.unmakeMove(m, undo)
			fmt.Printf("%v: %d\n", m, count)
			total += count
		}
	}
	fmt.Printf("\nTotal: %d\n", total)
}

func runSearchAndReport(p *Position, tc *TimeControl) {
	defer searchWG.Done()
	move := p.search(tc)
	if !currentTC.CompareAndSwap(tc, nil) {
		return
	}
	fmt.Println("bestmove", move)
}

// stopSearch halts any running search and waits for its goroutine to finish.
func stopSearch() {
	if cur := currentTC.Swap(nil); cur != nil {
		cur.Stop()
	}
	searchWG.Wait()
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

/*
  ----------------------------------------------------------------------------------
   UCI MAIN LOOP (Universal Chess Interface)
  ----------------------------------------------------------------------------------
   This is the communication part. The GUI (Arena, Banksia, Cutechess) sends text commands, we reply with text.

   [ GUI ] -------- "position startpos moves e2e4" ---->  [ ENGINE ]
   [ GUI ] <------- "info depth 5 score cp 20..." ------  [ ENGINE ]
   [ GUI ] -------- "go wtime 60000" ------------------>  [ ENGINE ]
   [ GUI ] <------- "bestmove e7e5" --------------------  [ ENGINE ]

   The loop waits for Stdin input, parses the string, and triggers engine functions.
   It must be non-blocking where possible to handle "stop" commands.
*/

func uciLoop() {
	pos := NewPosition()
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Fprintln(os.Stderr, "# Soomi V1.2.0B ready. Type 'help' for available commands.")

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		cmd := parts[0]

		switch cmd {
		case "uci":
			fmt.Println("id name Soomi V1.2.0B")
			fmt.Println("id author Otto Laukkanen")
			fmt.Println("option name Hash type spin default 256 min 1 max 4096")
			fmt.Println("uciok")

		case "isready":
			fmt.Println("readyok")

		case "setoption":
			name, value := parseSetOption(parts)
			if strings.EqualFold(name, "Hash") {
				sizeMB, err := strconv.Atoi(value)
				if err != nil || sizeMB <= 0 {
					fmt.Printf("info string invalid hash value: %s\n", value)
					continue
				}
				stopSearch()
				InitTT(sizeMB)
				fmt.Printf("info string Hash set to %d MB\n", sizeMB)
			} else {
				fmt.Printf("info string setoption %q = %q (ignored)\n", name, value)
			}

		case "ucinewgame":
			stopSearch()
			tt.Clear()
			clearHeuristics()
			pos.setStartPos()

		case "position":
			stopSearch()
			if len(parts) < 2 {
				fmt.Println("# Error: position requires arguments")
				continue
			}
			movesAt := len(parts)
			for i := 2; i < len(parts); i++ {
				if parts[i] == "moves" {
					movesAt = i
					break
				}
			}
			switch parts[1] {
			case "startpos":
				pos.setStartPos()
			case "fen":
				pos.setFEN(strings.Join(parts[2:movesAt], " "))
			}
			for _, mvStr := range parts[min(movesAt+1, len(parts)):] {
				if len(mvStr) < 4 || len(mvStr) > 5 {
					fmt.Printf("# Error: invalid move format: %s. Further moves ignored.\n", mvStr)
					break
				}
				found := false
				var buf [256]Move
				n := pos.generateMovesTo(buf[:], false)
				for j := 0; j < n; j++ {
					m := buf[j]
					if strings.EqualFold(m.String(), mvStr) && pos.isLegal(m) {
						pos.makeMove(m)
						found = true
						break
					}
				}
				if !found {
					fmt.Printf("# Error: illegal move: %s. Further moves ignored.\n", mvStr)
					break
				}
			}
		case "go":
			stopSearch()
			tc := &TimeControl{}
			for i := 1; i < len(parts); i++ {
				if parts[i] == "infinite" {
					tc.infinite = true
					continue
				}
				if i+1 == len(parts) {
					break
				}
				v, _ := strconv.ParseInt(parts[i+1], 10, 64)
				switch parts[i] {
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
				default:
					continue // unknown token: do not consume a value
				}
				i++ // skip the value just read
			}

			tc.allocateTime(pos.side)
			pcopy := *pos
			currentTC.Store(tc)

			searchWG.Add(1)
			go runSearchAndReport(&pcopy, tc)

		case "stop":
			if cur := currentTC.Load(); cur != nil {
				cur.Stop()
			}

		case "quit":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			return

		case "d", "display":
			fmt.Println("\n   a b c d e f g h")
			fmt.Println("  ----------------")
			for r := 7; r >= 0; r-- {
				fmt.Printf("%d|", r+1)
				for f := 0; f < 8; f++ {
					if v := pos.square[r*8+f]; v >= 0 {
						fmt.Printf(" %c", "PNBRQKpnbrqk"[(v>>3)*6+v&7])
					} else {
						fmt.Print(" .")
					}
				}
				fmt.Printf(" |%d\n", r+1)
			}
			fmt.Println("  ----------------")
			fmt.Println("   a b c d e f g h")
			fmt.Printf("Side to move: %s\n", [2]string{"White", "Black"}[pos.side])
			fmt.Printf("Hash: %x\n\n", pos.hash)

		case "eval":
			score := pos.evaluate()
			fmt.Printf("Evaluation: %+d (from %s's perspective)\n", score, [2]string{"White", "Black"}[pos.side])

		case "audit":
			fmt.Println("# Starting internal state audit...")
			// 1. Bitboard const
			occ := Bitboard(0)
			for c := 0; c < 2; c++ {
				for pt := 0; pt < 6; pt++ {
					occ |= pos.pieces[c][pt]
				}
			}
			if occ != pos.all {
				fmt.Printf("!! BITBOARD DESYNC: pos.all (%x) != calculated (%x)\n", pos.all, occ)
			} else {
				fmt.Println("  - Bitboard occupancy: OK")
			}

			// 2. Hash const
			expectedHash := uint64(0)
			if pos.side == Black {
				expectedHash ^= zobristSide
			}
			expectedHash ^= zobristCastleDiff[pos.castle]
			if pos.epSquare != -1 {
				expectedHash ^= zobristEP[pos.epSquare%8]
			}
			expectedPawnHash := uint64(0)
			for sq, v := range pos.square {
				if v >= 0 {
					expectedHash ^= zobristPiece[v>>3][v&7][sq]
					if v&7 == Pawn {
						expectedPawnHash ^= zobristPiece[v>>3][Pawn][sq]
					}
				}
			}
			if expectedHash != pos.hash {
				fmt.Printf("!! HASH DESYNC: pos.hash (%x) != calculated (%x)\n", pos.hash, expectedHash)
			} else {
				fmt.Println("  - Zobrist hash: OK")
			}
			if expectedPawnHash != pos.pawnHash {
				fmt.Printf("!! PAWN HASH DESYNC: pos.pawnHash (%x) != calculated (%x)\n", pos.pawnHash, expectedPawnHash)
			} else {
				fmt.Println("  - Pawn hash: OK")
			}
			fmt.Println("# Audit complete.")

		case "perft":
			if len(parts) > 1 {
				maxDepth, _ := strconv.Atoi(parts[1])
				fmt.Println("\nRunning perft test...")
				fmt.Println("Depth    Nodes           Time        NPS")
				fmt.Println("---------------------------------------------")
				for depth := 1; depth <= maxDepth; depth++ {
					start := time.Now()
					count := pos.perft(depth)
					elapsed := time.Since(start)
					nps := int64(0)
					if elapsed.Seconds() > 0 {
						nps = int64(float64(count) / elapsed.Seconds())
					}
					timeStr := fmt.Sprintf("%d ms", elapsed.Milliseconds())
					if elapsed >= time.Second {
						timeStr = fmt.Sprintf("%.2f s", elapsed.Seconds())
					}
					fmt.Printf("%-8d %-15d %-11s %d\n", depth, count, timeStr, nps)
				}
				fmt.Println()
			} else {
				fmt.Println("# Usage: perft <depth>")
			}

		case "divide":
			if len(parts) > 1 {
				depth, _ := strconv.Atoi(parts[1])
				pos.perftDivide(depth)
			} else {
				fmt.Println("# Usage: divide <depth>")
			}

		case "help":
			printHelp()

		default:
			fmt.Printf("# Unknown command: %s (type 'help' for available commands)\n", cmd)
		}
	}
}

func printHelp() {
	fmt.Println(`# Soomi V1.2.0B - Available Commands:

UCI Protocol Commands:
  uci                              - Initialize UCI mode
  isready                          - Check if engine is ready
  ucinewgame                       - Start a new game
  position startpos                - Set starting position
  position startpos moves <moves>  - Set position after moves
  go [options]                     - Start searching
      wtime <ms>                   - White's remaining time
      btime <ms>                   - Black's remaining time
      winc <ms>                    - White's increment per move
      binc <ms>                    - Black's increment per move
      movestogo <n>                - Moves until time control
      depth <n>                    - Search to fixed depth
      movetime <ms>                - Search for fixed time
      infinite                     - Search indefinitely
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
     d`)
}

func main() {
	fmt.Fprintln(os.Stderr, "Soomi V1.2.0B - UCI Chess Engine")
	fmt.Fprintln(os.Stderr, "Type 'help' for available commands or 'uci' to enter UCI mode")
	uciLoop()
}

// To make an executable
// set GOAMD64=v3 && go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi-V1.2.0B.exe Soomi.go
