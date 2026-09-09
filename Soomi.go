package main

import (
	"bufio"
	"fmt"
	"math"
	"math/bits"
	"os"
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
	defaultTTSizeMB                    = 256
	scoreHash                          = 1000000
	scorePromoBase                     = 900000
	scoreCaptureBase                   = 800000
	scoreKiller1                       = 750000
	scoreKiller2                       = 740000
	scoreCountermove                   = 200000
	minTimeMs            int64         = 5
	perMoveCapDiv        int64         = 3
	nextIterMult                       = 2
	continueMargin       time.Duration = 10 * time.Millisecond
	MaxGamePly                         = 1024
	ZobristSeed                        = 1070372
	totalPhase                         = 24
	contemptValue                      = -100
)

const (
	FlagQuiet         = 0
	FlagCapture       = 4
	FlagEP            = 5
	FlagCastle        = 2
	FlagPromoN        = 8
	FlagPromoB        = 9
	FlagPromoR        = 10
	FlagPromoQ        = 11
	FlagPromoCN       = 12
	FlagPromoCB       = 13
	FlagPromoCR       = 14
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
	pieceValues       = [6]int{100, 320, 300, 500, 900, 20000}
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
	passedPawnBonus   = [8]int{0, 10, 20, 40, 70, 110, 160, 0}
	mobilityBonus     = [4][]int{
		{-20, -10, 0, 10, 15, 20, 25, 30, 35},                                                                                  // Knight
		{-20, -10, 0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60},                                                              // Bishop
		{-20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60},                                                           // Rook
		{-40, -20, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125}, // Queen
	}
	kingZoneMask       [64]Bitboard
	kingAttackerWeight = [6]int{0, 2, 2, 3, 5, 0} // P, N, B, R, Q, K (P and K usually 0 or special)
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
	isolatedPawnPenalty   = 15
	doubledPawnPenalty    = 10
	bonusBishopPair       = 30
	bonusRookOpenFile     = 20
	bonusRookSemiOpenFile = 8
	bonusPawnShield       = 10
	penaltyPawnStorm      = 5
	bonusKnightOutpost    = 20
	penaltyKingTropism    = 2
	bonusRookOn7th        = 20
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

func (p *Position) computePhase() int {
	p.phase = totalPhase
	for pt := Knight; pt <= Queen; pt++ {
		p.phase -= bits.OnesCount64(uint64(p.pieces[White][pt]|p.pieces[Black][pt])) * piecePhase[pt]
	}
	return p.phase
}

func (p *Position) isEndgame() bool {
	return p.phase > 18
}

func rookMask(sq int) Bitboard {
	result := Bitboard(0)
	r, f := sq/8, sq%8

	for rr := r + 1; rr <= 6; rr++ {
		result |= Bitboard(1) << (rr*8 + f)
	}
	for rr := r - 1; rr >= 1; rr-- {
		result |= Bitboard(1) << (rr*8 + f)
	}

	for ff := f + 1; ff <= 6; ff++ {
		result |= Bitboard(1) << (r*8 + ff)
	}
	for ff := f - 1; ff >= 1; ff-- {
		result |= Bitboard(1) << (r*8 + ff)
	}

	return result
}

func bishopMask(sq int) Bitboard {
	result := Bitboard(0)
	r, f := sq/8, sq%8

	for rr, ff := r+1, f+1; rr <= 6 && ff <= 6; rr, ff = rr+1, ff+1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	for rr, ff := r-1, f+1; rr >= 1 && ff <= 6; rr, ff = rr-1, ff+1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	for rr, ff := r-1, f-1; rr >= 1 && ff >= 1; rr, ff = rr-1, ff-1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	for rr, ff := r+1, f-1; rr <= 6 && ff >= 1; rr, ff = rr+1, ff-1 {
		result |= Bitboard(1) << (rr*8 + ff)
	}

	return result
}

func occupancyVariations(mask Bitboard) []Bitboard {
	count := 1 << bits.OnesCount64(uint64(mask))
	variations := make([]Bitboard, 0, count)
	subset := Bitboard(0)
	for {
		variations = append(variations, subset)
		subset = (subset - mask) & mask
		if subset == 0 {
			break
		}
	}

	return variations
}

func magicIndex(occ Bitboard, magic Bitboard, shift uint8) uint32 {
	return uint32((occ * magic) >> shift)
}

func initMagicBitboards() {
	rookTableSize := 0
	bishopTableSize := 0

	for sq := 0; sq < 64; sq++ {
		mask := rookMask(sq)
		bitCount := bits.OnesCount64(uint64(mask))
		rookMagics[sq].mask = mask
		rookMagics[sq].magic = Bitboard(rookMagicNumbers[sq])
		rookMagics[sq].shift = 64 - uint8(bitCount)
		rookMagics[sq].offset = uint32(rookTableSize)
		rookTableSize += 1 << bitCount

		mask = bishopMask(sq)
		bitCount = bits.OnesCount64(uint64(mask))
		bishopMagics[sq].mask = mask
		bishopMagics[sq].magic = Bitboard(bishopMagicNumbers[sq])
		bishopMagics[sq].shift = 64 - uint8(bitCount)
		bishopMagics[sq].offset = uint32(bishopTableSize)
		bishopTableSize += 1 << bitCount
	}

	for sq := 0; sq < 64; sq++ {
		mask := rookMagics[sq].mask
		variations := occupancyVariations(mask)

		for _, occ := range variations {
			attacks := rookAttacksClassical(sq, occ)
			idx := magicIndex(occ, rookMagics[sq].magic, rookMagics[sq].shift)
			rookAttackTable[rookMagics[sq].offset+idx] = attacks
		}
	}

	for sq := 0; sq < 64; sq++ {
		mask := bishopMagics[sq].mask
		variations := occupancyVariations(mask)

		for _, occ := range variations {
			attacks := bishopAttacksClassical(sq, occ)
			idx := magicIndex(occ, bishopMagics[sq].magic, bishopMagics[sq].shift)
			bishopAttackTable[bishopMagics[sq].offset+idx] = attacks
		}
	}
}

func rookAttacksClassical(sq int, occ Bitboard) Bitboard {
	attacks := Bitboard(0)
	r, f := sq/8, sq%8

	for _, dr := range []int{1, -1} {
		for rr := r + dr; rr >= 0 && rr < 8; rr += dr {
			attacks |= Bitboard(1) << (rr*8 + f)
			if occ&(Bitboard(1)<<(rr*8+f)) != 0 {
				break
			}
		}
	}

	for _, df := range []int{1, -1} {
		for ff := f + df; ff >= 0 && ff < 8; ff += df {
			attacks |= Bitboard(1) << (r*8 + ff)
			if occ&(Bitboard(1)<<(r*8+ff)) != 0 {
				break
			}
		}
	}
	return attacks
}

func bishopAttacksClassical(sq int, occ Bitboard) Bitboard {
	attacks := Bitboard(0)
	r, f := sq/8, sq%8

	dirs := [][2]int{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
	for _, dir := range dirs {
		for rr, ff := r+dir[0], f+dir[1]; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr, ff = rr+dir[0], ff+dir[1] {
			attacks |= Bitboard(1) << (rr*8 + ff)
			if occ&(Bitboard(1)<<(rr*8+ff)) != 0 {
				break
			}
		}
	}
	return attacks
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

func (p *Position) isDraw() bool {
	return p.halfmove >= 100 || p.isRepetition() || p.isInsufficientMaterial()
}

func (p *Position) isInsufficientMaterial() bool {
	return (p.pieces[White][Pawn]|p.pieces[Black][Pawn]|p.pieces[White][Rook]|p.pieces[Black][Rook]|p.pieces[White][Queen]|p.pieces[Black][Queen]) == 0 && bits.OnesCount64(uint64(p.occupied[White])) <= 2 && bits.OnesCount64(uint64(p.occupied[Black])) <= 2
}

type ttEntry struct {
	key    uint64
	packed uint64
}

func packEntry(move uint32, score int16, gen uint8, depth uint8, flag uint8) uint64 {
	return uint64(move)<<32 | uint64(uint16(score))<<16 | uint64(gen)<<8 | uint64(depth)<<2 | uint64(flag&0x3)
}

func (e ttEntry) unpack() (move uint32, score int16, gen uint8, depth uint8, flag uint8) {
	p := e.packed
	move = uint32(p >> 32)
	score = int16(p >> 16)
	gen = uint8(p >> 8)
	depth = uint8((p >> 2) & 0x3F)
	flag = uint8(p & 0x3)
	return
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
	// A single ttEntry contains two uint64 variables (key and packed).
	// Each uint64 is 8 bytes, so one entry takes exactly 16 bytes of RAM.
	entrySize := uint64(16)

	// Convert the requested size from Megabytes (MB) into exact Bytes.
	// (1 MB = 1024 KB, 1 KB = 1024 Bytes).
	totalBytes := uint64(sizeMB) * 1024 * 1024

	// Calculate how much we can fit in that memory
	entries := totalBytes / entrySize

	// bits.Len64(entries) finds out how many bits it takes to represent in binary
	// Then by subtracting 1 and bitshifting by 1 we force the number to round down to a power of 2.
	// For example, if we have 100 entries, it goes down to 64.
	size := uint64(1) << (bits.Len64(entries) - 1)

	// Allocate table in memory
	tt = &TranspositionTable{
		// Create array with power of 2
		entries: make([]ttEntry, int(size)),
		// Due to size being calculated with our subtract and bitshift stuff,
		// Subtracting 1 from it gives us a binary mask that we can use
		// later with a AND instead of a modulo
		// as modulo is much slower than AND
		mask: size - 1,
	}
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

func (t *TranspositionTable) Probe(key uint64, minDepth int) (Move, int, uint8, int, bool, bool) {
	idx := int(key & t.mask)
	e := t.entries[idx]
	if e.key != key {
		return 0, 0, 0, 0, false, false
	}
	move, score, gen, depth, flag := e.unpack()
	if gen != uint8(t.gen) {
		return Move(move), int(score), flag, int(depth), true, false
	}
	return Move(move), int(score), flag, int(depth), true, int(depth) >= minDepth
}

// Save stores a search result into memory.
// We use always replace, so we can get relevant entries
// Its simple but can overwrite good entries with trash ones
func (t *TranspositionTable) Save(key uint64, mv Move, score int, depth int, flag uint8) {
	depth = min(depth, 63)
	idx := int(key & t.mask)
	newPacked := packEntry(uint32(mv), int16(score), uint8(t.gen), uint8(depth), flag)
	t.entries[idx] = ttEntry{key: key, packed: newPacked}
}

func (t *TranspositionTable) Hashfull() int {
	if t == nil || len(t.entries) == 0 {
		return 0
	}
	sampleSize := min(1000, len(t.entries))
	used := 0
	gen := uint8(t.gen)
	for i := 0; i < sampleSize; i++ {
		e := t.entries[i]
		if e.key != 0 && uint8(e.packed>>8) == gen {
			used++
		}
	}
	return (used * 1000) / sampleSize
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
		f := sq % 8
		r := sq / 8
		// White
		for nr := r + 1; nr < 8; nr++ {
			passedPawnMask[White][sq] |= sqBB[nr*8+f]
			if f > 0 {
				passedPawnMask[White][sq] |= sqBB[nr*8+f-1]
			}
			if f < 7 {
				passedPawnMask[White][sq] |= sqBB[nr*8+f+1]
			}
		}
		// Black
		for nr := r - 1; nr >= 0; nr-- {
			passedPawnMask[Black][sq] |= sqBB[nr*8+f]
			if f > 0 {
				passedPawnMask[Black][sq] |= sqBB[nr*8+f-1]
			}
			if f < 7 {
				passedPawnMask[Black][sq] |= sqBB[nr*8+f+1]
			}
		}
	}
}

func initPawnShield() {
	for sq := 0; sq < 64; sq++ {
		r, f := sq/8, sq%8
		// White
		if r <= 2 {
			fStart := max(0, f-1)
			fEnd := min(7, f+1)
			for nf := fStart; nf <= fEnd; nf++ {
				pawnShieldMask[White][sq] |= sqBB[(r+1)*8+nf]
				pawnShieldMask[White][sq] |= sqBB[(r+2)*8+nf]
			}
		}
		// Black
		if r >= 5 {
			fStart := max(0, f-1)
			fEnd := min(7, f+1)
			for nf := fStart; nf <= fEnd; nf++ {
				pawnShieldMask[Black][sq] |= sqBB[(r-1)*8+nf]
				pawnShieldMask[Black][sq] |= sqBB[(r-2)*8+nf]
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
		mask := kingAttacks[sq] | sqBB[sq]
		r := sq / 8
		// Include squares in front
		if r < 7 {
			mask |= (mask << 8)
		}
		if r > 0 {
			mask |= (mask >> 8)
		}
		kingZoneMask[sq] = mask
	}
}

func initPST() {
	pst[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		-5, 10, 10, -20, -20, 10, 10, -5,
		0, 0, -10, 5, 5, 0, 0, 0,
		0, -10, 10, 20, 20, 10, 5, 0,
		10, 10, 15, 25, 25, 15, 10, 10,
		15, 15, 20, 30, 30, 20, 15, 15,
		30, 30, 30, 40, 40, 30, 30, 30,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pst[White][Knight] = [64]int{
		-30, -20, -10, -10, -10, -10, -20, -30,
		-20, -10, 5, 5, 5, 5, -10, -20,
		-20, 5, 15, 15, 15, 15, 5, -20,
		-10, 5, 15, 20, 20, 15, 5, -10,
		-10, 5, 15, 25, 25, 15, 5, -10,
		-20, 5, 10, 15, 15, 10, 5, -20,
		-20, 0, 0, 0, 0, 0, 0, -20,
		-30, -10, -10, -10, -10, -10, -20, -30,
	}

	pst[White][Bishop] = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 10, 5, 5, 5, 5, 10, -10,
		-10, 5, 5, 15, 15, 5, 5, -10,
		-10, 5, 5, 15, 15, 5, 5, -10,
		-10, 5, 10, 20, 20, 10, 5, -10,
		-10, 10, 10, 15, 15, 10, 10, -10,
		-10, 10, 5, 5, 5, 5, 10, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}

	pst[White][Rook] = [64]int{
		0, 0, 5, 10, 10, 5, 0, 0,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		10, 15, 15, 20, 20, 15, 15, 10,
		0, 0, 0, 5, 5, 0, 0, 0,
	}

	pst[White][Queen] = [64]int{
		-20, -10, -10, -5, -5, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 5, 5, 5, 5, 0, -10,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-10, 5, 5, 5, 5, 5, 5, -10,
		-10, 0, 5, 5, 5, 5, 0, -10,
		-20, -10, -10, -5, -5, -10, -10, -20,
	}

	pst[White][King] = [64]int{
		30, 20, 5, -10, -10, 5, 20, 30,
		10, 10, -15, -30, -30, -15, 10, 10,
		-20, -20, -20, -20, -20, -20, -20, -20,
		-20, -30, -30, -40, -40, -30, -30, -20,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
	}

	pstEnd[White][Pawn] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		10, 10, 10, 10, 10, 10, 10, 10,
		20, 20, 20, 20, 20, 20, 20, 20,
		30, 30, 30, 30, 30, 30, 30, 30,
		40, 40, 40, 40, 40, 40, 40, 40,
		60, 60, 60, 60, 60, 60, 60, 60,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	pstEnd[White][Knight] = [64]int{
		-20, -10, -5, -5, -5, -5, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 5, 5, 5, 5, 5, 5, -10,
		-5, 5, 5, 10, 10, 5, 5, -5,
		-5, 5, 5, 10, 10, 5, 5, -5,
		-10, 5, 5, 5, 5, 5, 5, -10,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-20, -10, -5, -5, -5, -5, -10, -20,
	}

	pstEnd[White][Bishop] = [64]int{
		-10, -5, -5, -5, -5, -5, -5, -10,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 5, 5, 5, 5, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-10, -5, -5, -5, -5, -5, -5, -10,
	}

	pstEnd[White][Rook] = [64]int{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		15, 20, 20, 25, 25, 20, 20, 15,
		10, 10, 10, 10, 10, 10, 10, 10,
	}

	pstEnd[White][King] = [64]int{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 10, 20, 20, 10, 0, -10,
		-10, 0, 10, 30, 30, 10, 0, -10,
		-10, 0, 10, 30, 30, 10, 0, -10,
		-10, 0, 10, 20, 20, 10, 0, -10,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
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
	wk, wq := next(), next()
	bk, bq := next(), next()
	for i := 0; i < 8; i++ {
		zobristEP[i] = next()
	}
	// Precompute XOR for makemove
	for i := 0; i < 16; i++ {
		if i&1 != 0 {
			zobristCastleDiff[i] ^= wk
		}
		if i&2 != 0 {
			zobristCastleDiff[i] ^= wq
		}
		if i&4 != 0 {
			zobristCastleDiff[i] ^= bk
		}
		if i&8 != 0 {
			zobristCastleDiff[i] ^= bq
		}
	}
}

func initAttacks() {
	knightDirs := []int{-17, -15, -10, -6, 6, 10, 15, 17}
	kingDirs := []int{-9, -8, -7, -1, 1, 7, 8, 9}

	for sq := 0; sq < 64; sq++ {
		r, f := sq>>3, sq&7

		for _, d := range knightDirs {
			to := sq + d
			if to >= 0 && to < 64 {
				tr, tf := to>>3, to&7
				if abs(r-tr) <= 2 && abs(f-tf) <= 2 {
					knightAttacks[sq] |= sqBB[to]
				}
			}
		}

		for _, d := range kingDirs {
			to := sq + d
			if to >= 0 && to < 64 {
				tr, tf := to>>3, to&7
				if abs(r-tr) <= 1 && abs(f-tf) <= 1 {
					kingAttacks[sq] |= sqBB[to]
				}
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
			switch ch {
			case 'K':
				p.castle |= 1
			case 'Q':
				p.castle |= 2
			case 'k':
				p.castle |= 4
			case 'q':
				p.castle |= 8
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

func (p *Position) pieceAt(sq int) (color, piece int, ok bool) {
	val := p.square[sq]
	if val < 0 {
		return 0, 0, false
	}
	color = val >> 3
	piece = val & 7
	return color, piece, true
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

	occAll := p.all
	occUs := p.occupied[us]
	occThem := p.occupied[them]
	ep := p.epSquare
	pawns := p.pieces[us][Pawn]
	var push, dblPush int
	var promoRank, dblRank int
	if us == White {
		push, dblPush = 8, 16
		promoRank, dblRank = 6, 1
	} else {
		push, dblPush = -8, -16
		promoRank, dblRank = 1, 6
	}

	for bb := pawns; bb != 0; {
		from := popLSB(&bb)
		to := from + push

		if from>>3 == promoRank {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				buf[i] = makeMoves(from, to, FlagPromoQ)
				i++
				buf[i] = makeMoves(from, to, FlagPromoR)
				i++
				buf[i] = makeMoves(from, to, FlagPromoB)
				i++
				buf[i] = makeMoves(from, to, FlagPromoN)
				i++
			}

			attacks := pawnAttacks[us][from] & occThem
			for att := attacks; att != 0; {
				to := popLSB(&att)
				buf[i] = makeMoves(from, to, FlagPromoCQ)
				i++
				buf[i] = makeMoves(from, to, FlagPromoCR)
				i++
				buf[i] = makeMoves(from, to, FlagPromoCB)
				i++
				buf[i] = makeMoves(from, to, FlagPromoCN)
				i++
			}
		} else {
			if !capturesOnly && occAll&sqBB[to] == 0 {
				buf[i] = makeMoves(from, to, FlagQuiet)
				i++

				if from>>3 == dblRank {
					to2 := from + dblPush
					if occAll&sqBB[to2] == 0 {
						buf[i] = makeMoves(from, to2, FlagQuiet)
						i++
					}
				}
			}

			attacks := pawnAttacks[us][from] & occThem
			for att := attacks; att != 0; {
				to := popLSB(&att)
				buf[i] = makeMoves(from, to, FlagCapture)
				i++
			}

			if ep >= 0 {
				if pawnAttacks[us][from]&sqBB[ep] != 0 {
					buf[i] = makeMoves(from, ep, FlagEP)
					i++
				}
			}
		}
	}

	for bb := p.pieces[us][Knight]; bb != 0; {
		from := popLSB(&bb)
		attacks := knightAttacks[from] & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	for bb := p.pieces[us][Bishop]; bb != 0; {
		from := popLSB(&bb)
		attacks := bishopAttacks(from, occAll) & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	for bb := p.pieces[us][Rook]; bb != 0; {
		from := popLSB(&bb)
		attacks := rookAttacks(from, occAll) & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	for bb := p.pieces[us][Queen]; bb != 0; {
		from := popLSB(&bb)
		attacks := (bishopAttacks(from, occAll) | rookAttacks(from, occAll)) & ^occUs
		if capturesOnly {
			attacks &= occThem
		}
		for att := attacks; att != 0; {
			to := popLSB(&att)
			flag := FlagQuiet
			if occThem&sqBB[to] != 0 {
				flag = FlagCapture
			}
			buf[i] = makeMoves(from, to, flag)
			i++
		}
	}

	// da king moves
	kingSq := p.kingSq[us]
	attacks := kingAttacks[kingSq] & ^occUs
	if capturesOnly {
		attacks &= occThem
	}
	for att := attacks; att != 0; {
		to := popLSB(&att)
		flag := FlagQuiet
		if (occThem & sqBB[to]) != 0 {
			flag = FlagCapture
		}
		buf[i] = makeMoves(kingSq, to, flag)
		i++
	}

	if !capturesOnly && !p.inCheck() {
		if us == White {
			if p.castle&1 != 0 && occAll&0x60 == 0 {
				buf[i] = makeMoves(4, 6, FlagCastle)
				i++
			}
			if p.castle&2 != 0 && occAll&0x0E == 0 {
				buf[i] = makeMoves(4, 2, FlagCastle)
				i++
			}
		} else {
			if p.castle&4 != 0 && occAll&(0x60<<56) == 0 {
				buf[i] = makeMoves(60, 62, FlagCastle)
				i++
			}
			if p.castle&8 != 0 && occAll&(0x0E<<56) == 0 {
				buf[i] = makeMoves(60, 58, FlagCastle)
				i++
			}
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
	h := p.hash
	h ^= zobristSide

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

		p.halfmove = 0
	} else if movingPiece == Pawn {
		p.halfmove = 0
	} else {
		p.halfmove++
	}

	if flags == FlagCastle {
		bbFrom := sqBB[from]
		bbTo := sqBB[to]
		p.pieces[us][King] &^= bbFrom
		p.pieces[us][King] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
		h ^= zobristPiece[us][King][from] ^ zobristPiece[us][King][to]
		p.psqScore[us] += pst[us][King][to] - pst[us][King][from]
		p.psqScoreEG[us] += pstEnd[us][King][to] - pstEnd[us][King][from]
		p.square[from] = -1
		p.square[to] = (us << 3) | King
		p.kingSq[us] = to
		var rf, rt int
		if to > from {
			rf, rt = from+3, from+1
		} else {
			rf, rt = from-4, from-1
		}
		rfBB := sqBB[rf]
		rtBB := sqBB[rt]
		p.pieces[us][Rook] &^= rfBB
		p.pieces[us][Rook] |= rtBB
		p.occupied[us] &^= rfBB
		p.occupied[us] |= rtBB
		p.all &^= rfBB
		p.all |= rtBB
		h ^= zobristPiece[us][Rook][rf] ^ zobristPiece[us][Rook][rt]
		p.psqScore[us] += pst[us][Rook][rt] - pst[us][Rook][rf]
		p.psqScoreEG[us] += pstEnd[us][Rook][rt] - pstEnd[us][Rook][rf]
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		bbFrom := sqBB[from]
		p.pieces[us][Pawn] &^= bbFrom
		p.occupied[us] &^= bbFrom
		p.all &^= bbFrom
		h ^= zobristPiece[us][Pawn][from]
		p.pawnHash ^= zobristPiece[us][Pawn][from]
		p.material[us] -= pieceValues[Pawn]
		p.psqScore[us] -= pst[us][Pawn][from]
		p.psqScoreEG[us] -= pstEnd[us][Pawn][from]
		p.square[from] = -1

		promoType := (flags & 3) + Knight
		bbTo := sqBB[to]
		p.pieces[us][promoType] |= bbTo
		p.occupied[us] |= bbTo
		p.all |= bbTo
		h ^= zobristPiece[us][promoType][to]

		p.material[us] += pieceValues[promoType]
		p.phase -= piecePhase[promoType]
		p.psqScore[us] += pst[us][promoType][to]
		p.psqScoreEG[us] += pstEnd[us][promoType][to]
		p.square[to] = (us << 3) | promoType

	} else {
		bbFrom := sqBB[from]
		bbTo := sqBB[to]
		p.pieces[us][movingPiece] &^= bbFrom
		p.pieces[us][movingPiece] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
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
	irreversible := (undo.captured >= 0) || (movingPiece == Pawn)
	p.historyPly++
	p.historyKeys[p.historyPly] = h

	if irreversible {
		p.lastIrreversible = p.historyPly
	}
	p.hash = h
	return undo
}

func (p *Position) unmakeMove(m Move, undo Undo) {
	from, to, flags := m.from(), m.to(), m.flags()
	us := p.side ^ 1
	them := p.side
	p.historyPly--
	p.lastIrreversible = undo.lastIrreversible
	p.side = us
	p.castle = undo.castle
	p.epSquare = undo.epSquare
	p.halfmove = undo.halfmove

	if flags == FlagCastle {
		bbFrom := sqBB[to]
		bbTo := sqBB[from]
		p.pieces[us][King] &^= bbFrom
		p.pieces[us][King] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
		p.psqScore[us] += pst[us][King][from] - pst[us][King][to]
		p.psqScoreEG[us] += pstEnd[us][King][from] - pstEnd[us][King][to]
		p.square[to] = -1
		p.square[from] = (us << 3) | King
		p.kingSq[us] = from

		var rf, rt int
		if to > from {
			rf, rt = from+1, from+3
		} else {
			rf, rt = from-1, from-4
		}

		rfBB := sqBB[rf]
		rtBB := sqBB[rt]
		p.pieces[us][Rook] &^= rfBB
		p.pieces[us][Rook] |= rtBB
		p.occupied[us] &^= rfBB
		p.occupied[us] |= rtBB
		p.all &^= rfBB
		p.all |= rtBB
		p.psqScore[us] += pst[us][Rook][rt] - pst[us][Rook][rf]
		p.psqScoreEG[us] += pstEnd[us][Rook][rt] - pstEnd[us][Rook][rf]
		p.square[rf] = -1
		p.square[rt] = (us << 3) | Rook

	} else if flags >= FlagPromoN {
		promoType := (flags & 3) + Knight
		bbTo := sqBB[to]
		p.pieces[us][promoType] &^= bbTo
		p.occupied[us] &^= bbTo
		p.all &^= bbTo
		p.material[us] -= pieceValues[promoType]
		p.psqScore[us] -= pst[us][promoType][to]
		p.psqScoreEG[us] -= pstEnd[us][promoType][to]
		p.phase += piecePhase[promoType]
		p.square[to] = -1
		bbFrom := sqBB[from]
		p.pieces[us][Pawn] |= bbFrom
		p.occupied[us] |= bbFrom
		p.all |= bbFrom
		p.material[us] += pieceValues[Pawn]
		p.psqScore[us] += pst[us][Pawn][from]
		p.psqScoreEG[us] += pstEnd[us][Pawn][from]
		p.square[from] = (us << 3) | Pawn
		p.pawnHash ^= zobristPiece[us][Pawn][from]

	} else {
		movingPt := p.square[to] & 7
		bbFrom := sqBB[to]
		bbTo := sqBB[from]
		p.pieces[us][movingPt] &^= bbFrom
		p.pieces[us][movingPt] |= bbTo
		p.occupied[us] &^= bbFrom
		p.occupied[us] |= bbTo
		p.all &^= bbFrom
		p.all |= bbTo
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

func (p *Position) see(m Move) int {
	from, to := m.from(), m.to()
	piece := p.square[from] & 7
	flags := m.flags()
	gain0 := 0
	if flags == FlagEP {
		gain0 = pieceValues[Pawn]
	} else if p.square[to] != -1 {
		gain0 = pieceValues[p.square[to]&7]
	}
	pieceAfter := piece
	if m.isPromo() {
		pieceAfter = m.promoType()
	}
	return p.seeIterative(from, to, pieceAfter, gain0)
}

func (p *Position) seeIterative(from, to, pieceAfterFirst, gain0 int) int {
	var gain [32]int
	d := 0
	// Pre-calculate slider masks
	diagSliders := p.pieces[White][Bishop] | p.pieces[Black][Bishop] | p.pieces[White][Queen] | p.pieces[Black][Queen]
	orthSliders := p.pieces[White][Rook] | p.pieces[Black][Rook] | p.pieces[White][Queen] | p.pieces[Black][Queen]

	occ := p.all
	att := p.getSEEAttackers(to, occ, diagSliders, orthSliders)

	// First move (the move m passed to see)
	us := p.side
	gain[d] = gain0
	piece := pieceAfterFirst

	att &^= sqBB[from]
	occ &^= sqBB[from]
	// Any move can uncover X-ray sliders
	att |= p.getXrayAttackers(to, occ, diagSliders, orthSliders)
	us ^= 1
	for {
		d++
		myAtt := att & p.occupied[us]
		var pt int
		var attSq int
		found := false
		// Find smallest attacker for 'us'
		for _, pType := range lvaOrder {
			subset := myAtt & p.pieces[us][pType]
			if subset != 0 {
				pt = pType
				attSq = bits.TrailingZeros64(uint64(subset))
				found = true
				break
			}
		}
		if !found {
			break
		}

		gain[d] = pieceValues[piece] - gain[d-1]
		piece = pt
		att &^= sqBB[attSq]
		occ &^= sqBB[attSq]
		att |= p.getXrayAttackers(to, occ, diagSliders, orthSliders)
		us ^= 1
	}

	for d--; d > 0; d-- {
		gain[d-1] = -max(-gain[d-1], gain[d])
	}
	return gain[0]
}

func (p *Position) getSEEAttackers(sq int, occ, diagSliders, orthSliders Bitboard) Bitboard {
	return (pawnAttacks[White][sq] & p.pieces[Black][Pawn] & occ) |
		(pawnAttacks[Black][sq] & p.pieces[White][Pawn] & occ) |
		(knightAttacks[sq] & (p.pieces[White][Knight] | p.pieces[Black][Knight]) & occ) |
		(kingAttacks[sq] & (p.pieces[White][King] | p.pieces[Black][King]) & occ) |
		(bishopAttacks(sq, occ) & diagSliders & occ) |
		(rookAttacks(sq, occ) & orthSliders & occ)
}

func (p *Position) getXrayAttackers(sq int, occ, diagSliders, orthSliders Bitboard) Bitboard {
	return (bishopAttacks(sq, occ) & diagSliders & occ) | (rookAttacks(sq, occ) & orthSliders & occ)
}

func updateHistory(side, from, to, bonus int) {
	clampedBonus := bonus
	if clampedBonus > 400 {
		clampedBonus = 400
	} else if clampedBonus < -400 {
		clampedBonus = -400
	}
	history[side][from][to] += clampedBonus - (history[side][from][to] * abs(clampedBonus) / MaxHistory)
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

	// We dont have it, recalculate
	mg, eg = 0, 0
	whitePawns := p.pieces[White][Pawn]
	blackPawns := p.pieces[Black][Pawn]

	// what pawns
	for bb := whitePawns; bb != 0; {
		sq := popLSB(&bb)
		file := sq % 8
		rank := sq / 8

		// Doubled
		if (whitePawns&fileMasks[file]) & ^sqBB[sq] != 0 {
			mg -= doubledPawnPenalty
			eg -= doubledPawnPenalty
		}

		// Isolated
		isolated := true
		if file > 0 && (whitePawns&fileMasks[file-1]) != 0 {
			isolated = false
		}
		if file < 7 && (whitePawns&fileMasks[file+1]) != 0 {
			isolated = false
		}
		if isolated {
			mg -= isolatedPawnPenalty
			eg -= isolatedPawnPenalty
		}

		// Passed
		if (blackPawns & passedPawnMask[White][sq]) == 0 {
			mg += passedPawnBonus[rank] / 2
			eg += passedPawnBonus[rank]
		}
	}

	// Blik pawns
	for bb := blackPawns; bb != 0; {
		sq := popLSB(&bb)
		file := sq % 8
		rank := sq / 8
		revRank := 7 - rank

		// Doubled
		if (blackPawns&fileMasks[file]) & ^sqBB[sq] != 0 {
			mg += doubledPawnPenalty
			eg += doubledPawnPenalty
		}

		// Isolated
		isolated := true
		if file > 0 && (blackPawns&fileMasks[file-1]) != 0 {
			isolated = false
		}
		if file < 7 && (blackPawns&fileMasks[file+1]) != 0 {
			isolated = false
		}
		if isolated {
			mg += isolatedPawnPenalty
			eg += isolatedPawnPenalty
		}

		// Passed
		if (whitePawns & passedPawnMask[Black][sq]) == 0 {
			mg -= passedPawnBonus[revRank] / 2
			eg -= passedPawnBonus[revRank]
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
	mg, eg = 0, 0
	occupied := p.all
	dangerW := ((p.pieces[Black][Pawn] & ^Bitboard(0x8080808080808080)) >> 7) | ((p.pieces[Black][Pawn] & ^Bitboard(0x0101010101010101)) >> 9)
	dangerB := ((p.pieces[White][Pawn] & ^Bitboard(0x0101010101010101)) << 7) | ((p.pieces[White][Pawn] & ^Bitboard(0x8080808080808080)) << 9)

	// What mobility
	for bb := p.pieces[White][Knight]; bb != 0; {
		sq := popLSB(&bb)
		attacks := knightAttacks[sq] & ^p.occupied[White] & ^dangerW
		cnt := bits.OnesCount64(uint64(attacks))
		mg += mobilityBonus[0][cnt]
		eg += mobilityBonus[0][cnt]
	}
	for bb := p.pieces[White][Bishop]; bb != 0; {
		sq := popLSB(&bb)
		attacks := bishopAttacks(sq, occupied) & ^p.occupied[White] & ^dangerW
		cnt := bits.OnesCount64(uint64(attacks))
		mg += mobilityBonus[1][cnt]
		eg += mobilityBonus[1][cnt]
	}
	for bb := p.pieces[White][Rook]; bb != 0; {
		sq := popLSB(&bb)
		attacks := rookAttacks(sq, occupied) & ^p.occupied[White] & ^dangerW
		cnt := bits.OnesCount64(uint64(attacks))
		mg += mobilityBonus[2][cnt]
		eg += mobilityBonus[2][cnt]
	}
	for bb := p.pieces[White][Queen]; bb != 0; {
		sq := popLSB(&bb)
		attacks := (bishopAttacks(sq, occupied) | rookAttacks(sq, occupied)) & ^p.occupied[White] & ^dangerW
		cnt := bits.OnesCount64(uint64(attacks))
		mg += mobilityBonus[3][cnt]
		eg += mobilityBonus[3][cnt]
	}

	// Blik mobility
	for bb := p.pieces[Black][Knight]; bb != 0; {
		sq := popLSB(&bb)
		attacks := knightAttacks[sq] & ^p.occupied[Black] & ^dangerB
		cnt := bits.OnesCount64(uint64(attacks))
		mg -= mobilityBonus[0][cnt]
		eg -= mobilityBonus[0][cnt]
	}
	for bb := p.pieces[Black][Bishop]; bb != 0; {
		sq := popLSB(&bb)
		attacks := bishopAttacks(sq, occupied) & ^p.occupied[Black] & ^dangerB
		cnt := bits.OnesCount64(uint64(attacks))
		mg -= mobilityBonus[1][cnt]
		eg -= mobilityBonus[1][cnt]
	}
	for bb := p.pieces[Black][Rook]; bb != 0; {
		sq := popLSB(&bb)
		attacks := rookAttacks(sq, occupied) & ^p.occupied[Black] & ^dangerB
		cnt := bits.OnesCount64(uint64(attacks))
		mg -= mobilityBonus[2][cnt]
		eg -= mobilityBonus[2][cnt]
	}
	for bb := p.pieces[Black][Queen]; bb != 0; {
		sq := popLSB(&bb)
		attacks := (bishopAttacks(sq, occupied) | rookAttacks(sq, occupied)) & ^p.occupied[Black] & ^dangerB
		cnt := bits.OnesCount64(uint64(attacks))
		mg -= mobilityBonus[3][cnt]
		eg -= mobilityBonus[3][cnt]
	}
	return mg, eg
}

func (p *Position) evalKingSafety() int {
	mg := 0
	whiteKingSq := p.kingSq[White]
	blackKingSq := p.kingSq[Black]

	zone := kingZoneMask[whiteKingSq]
	attackers := 0
	attackUnits := 0
	for pt := Knight; pt <= Queen; pt++ {
		bb := p.pieces[Black][pt]
		for bb != 0 {
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
	if attackers > 1 {
		mg -= (attackUnits * attackUnits)
	}
	// Pawn shield
	mg += p.evalPawnShield(White, whiteKingSq)

	zone = kingZoneMask[blackKingSq]
	attackers = 0
	attackUnits = 0
	for pt := Knight; pt <= Queen; pt++ {
		bb := p.pieces[White][pt]
		for bb != 0 {
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
	if attackers > 1 {
		mg += (attackUnits * attackUnits)
	}
	// Pawn shield
	mg -= p.evalPawnShield(Black, blackKingSq)

	return mg
}

func (p *Position) evalPawnShield(side int, kingSq int) int {
	pawns := p.pieces[side][Pawn]
	return bits.OnesCount64(uint64(pawns&pawnShieldMask[side][kingSq])) * bonusPawnShield
}

func (p *Position) evalPawnStorm() int {
	score := 0
	whiteKingSq := p.kingSq[White]
	blackKingSq := p.kingSq[Black]

	for bb := p.pieces[Black][Pawn] & passedPawnMask[White][whiteKingSq]; bb != 0; {
		sq := popLSB(&bb)
		dist := (sq / 8) - (whiteKingSq / 8)
		score -= penaltyPawnStorm * (6 - dist)
	}

	for bb := p.pieces[White][Pawn] & passedPawnMask[Black][blackKingSq]; bb != 0; {
		sq := popLSB(&bb)
		dist := (blackKingSq / 8) - (sq / 8)
		score += penaltyPawnStorm * (6 - dist)
	}
	return score
}

func (p *Position) evalOutposts() (mg, eg int) {
	mg, eg = 0, 0
	// Knight and Bishop outposts
	for side := White; side <= Black; side++ {
		enemy := side ^ 1
		enemyPawns := p.pieces[enemy][Pawn]
		for _, pt := range []int{Knight, Bishop} {
			for bb := p.pieces[side][pt]; bb != 0; {
				sq := popLSB(&bb)
				r, f := sq/8, sq%8

				// Outpost range
				if (side == White && r >= 3 && r <= 5) || (side == Black && r >= 2 && r <= 4) {
					// Check for pawn support (aww)
					if (pawnAttacks[side^1][sq] & p.pieces[side][Pawn]) != 0 {
						// Use precomputed masks
						attackMask := passedPawnMask[side][sq] &^ fileMasks[f]
						if (enemyPawns & attackMask) == 0 {
							bonus := bonusKnightOutpost
							if pt == Bishop {
								bonus /= 2
							}
							if side == White {
								mg += bonus
								eg += bonus / 2
							} else {
								mg -= bonus
								eg -= bonus / 2
							}
						}
					}
				}
			}
		}
	}
	return mg, eg
}

func (p *Position) evalTropism() int {
	score := 0
	whiteKingSq := p.kingSq[White]
	blackKingSq := p.kingSq[Black]

	wr, wf := whiteKingSq/8, whiteKingSq%8
	attackers := p.occupied[Black] &^ (p.pieces[Black][Pawn] | p.pieces[Black][King])
	for attackers != 0 {
		sq := popLSB(&attackers)
		dist := abs(sq/8-wr) + abs(sq%8-wf)
		score -= penaltyKingTropism * (14 - dist)
	}

	br, bf := blackKingSq/8, blackKingSq%8
	attackers = p.occupied[White] &^ (p.pieces[White][Pawn] | p.pieces[White][King])
	for attackers != 0 {
		sq := popLSB(&attackers)
		dist := abs(sq/8-br) + abs(sq%8-bf)
		score += penaltyKingTropism * (14 - dist)
	}

	return score
}

func (p *Position) evalRooksOnFiles() int {
	score := 0
	wPawns := p.pieces[White][Pawn]
	bPawns := p.pieces[Black][Pawn]

	for bb := p.pieces[White][Rook]; bb != 0; {
		sq := popLSB(&bb)
		f := sq % 8
		if (wPawns & fileMasks[f]) == 0 {
			if (bPawns & fileMasks[f]) == 0 {
				score += bonusRookOpenFile
			} else {
				score += bonusRookSemiOpenFile
			}
		}
		if (sq/8) == 6 && p.kingSq[Black] >= 56 {
			score += bonusRookOn7th
		}
	}

	for bb := p.pieces[Black][Rook]; bb != 0; {
		sq := popLSB(&bb)
		f := sq % 8
		if (bPawns & fileMasks[f]) == 0 {
			if (wPawns & fileMasks[f]) == 0 {
				score -= bonusRookOpenFile
			} else {
				score -= bonusRookSemiOpenFile
			}
		}
		if (sq/8) == 1 && p.kingSq[White] <= 7 {
			score -= bonusRookOn7th
		}
	}
	return score
}

func (p *Position) evaluate() int {
	a := p.material[White] - p.material[Black]
	b := p.psqScore[White] - p.psqScore[Black]
	c := p.psqScoreEG[White] - p.psqScoreEG[Black]

	mgScore := a + b
	egScore := a + c

	// Add Pawn Structure
	pawnMG, pawnEG := p.evalPawns()
	mgScore += pawnMG
	egScore += pawnEG

	// Add Mobility
	mobilityMG, mobilityEG := p.evalMobility()
	mgScore += mobilityMG
	egScore += mobilityEG

	// Add King Safety
	mgScore += p.evalKingSafety()
	mgScore += p.evalTropism()
	mgScore += p.evalPawnStorm()

	// Add Outposts
	outpostMG, outpostEG := p.evalOutposts()
	mgScore += outpostMG
	egScore += outpostEG

	// Add Bishop Pair
	bp := p.evalBishopPair()
	mgScore += bp
	egScore += bp

	// Add Rooks on Files
	rof := p.evalRooksOnFiles()
	mgScore += rof
	egScore += rof

	// Tempo
	// Calculation uses p.side which is 0 for white and 1 for black
	// Therefore for white 20 - 40 * p.side = 20 (as it should, p.side is 0)
	// And for black the calculation 20 - 40 * p.side gives = -20 (p.side is 1)
	mgScore += 20 - 40*p.side
	ph := p.phase
	phaseScaled := ((totalPhase-ph)*PhaseScale + totalPhase/2) / totalPhase
	score := egScore + ((mgScore-egScore)*phaseScaled)/PhaseScale
	// Same trick as tempo
	return score * (1 - 2*p.side)
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
	n := len(moves)
	var stackScores [256]int
	scores := stackScores[:n]
	for i := 0; i < n; i++ {
		m := moves[i]
		score := 0
		if m == bestMove {
			score = scoreHash
		} else if m.isPromo() {
			score = scorePromoBase + pieceValues[m.promoType()]
		} else if m.isCapture() {
			seeVal := p.see(m)
			if seeVal >= 0 {
				victim := Pawn
				if vt := p.square[m.to()]; vt != -1 {
					victim = vt & 7
				}
				score = scoreCaptureBase + seeVal + pieceValues[victim]*MVVLVAWeight - pieceValues[p.square[m.from()]&7]
			} else {
				// Bad
				score = seeVal
			}
		} else {
			switch m {
			case killer1:
				score = scoreKiller1
			case killer2:
				score = scoreKiller2
			default:
				us := p.side
				from, to := m.from(), m.to()
				pt := p.square[from] & 7
				score = history[us][from][to]
				if prevMove != 0 && m == countermoves[us][prevMove.from()][prevMove.to()] {
					// Countermove bonus
					score += scoreCountermove
				}
				// PST bonus for moves without history
				score += pst[us][pt][to] - pst[us][pt][from]
			}
		}
		scores[i] = score
	}

	for i := 1; i < n; i++ {
		kMove := moves[i]
		kScore := scores[i]
		j := i - 1
		for j >= 0 && scores[j] < kScore {
			moves[j+1] = moves[j]
			scores[j+1] = scores[j]
			j--
		}
		moves[j+1] = kMove
		scores[j+1] = kScore
	}

	return moves
}

func (p *Position) orderMovesQ(moves []Move, scores []int) {
	n := len(moves)
	for i := 0; i < n; i++ {
		m := moves[i]
		score := 0
		if m.isPromo() {
			score = scorePromoBase + pieceValues[m.promoType()]
		} else if m.isCapture() {
			seeVal := p.see(m)
			if seeVal >= 0 {
				victim := Pawn
				if vt := p.square[m.to()]; vt != -1 {
					victim = vt & 7
				}
				score = scoreCaptureBase + seeVal + pieceValues[victim]*MVVLVAWeight - pieceValues[p.square[m.from()]&7]
			} else {
				// Bad, Prune or sort later
				score = seeVal
			}
		}
		scores[i] = score
	}

	for i := 1; i < n; i++ {
		kMove := moves[i]
		kScore := scores[i]
		j := i - 1
		for j >= 0 && scores[j] < kScore {
			moves[j+1] = moves[j]
			scores[j+1] = scores[j]
			j--
		}
		moves[j+1] = kMove
		scores[j+1] = kScore
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
	if ply > p.seldepth {
		p.seldepth = ply
	}
	if ply >= MaxDepth {
		return p.evaluate()
	}
	p.localNodes++

	if (p.localNodes & NodeCheckMaskSearch) == 0 {
		if tc.shouldStop() {
			return alpha
		}
	}

	inCheck := p.inCheck()
	best := alpha
	if !inCheck {
		stand := p.evaluate()
		if stand >= beta {
			return stand
		}
		best = stand
		if stand > alpha {
			alpha = stand
		}
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
		if score > best {
			best = score
		}
		if score > alpha {
			alpha = score
		}
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

func (p *Position) negamax(depth, alpha, beta, ply int, pv *[]Move, tc *TimeControl, ss *[MaxDepth]SearchStack, prevMove Move) int {
	if ply > p.seldepth {
		p.seldepth = ply
	}
	if ply >= MaxDepth {
		return p.evaluate()
	}
	p.localNodes++

	// Time check
	if (p.localNodes & NodeCheckMaskSearch) == 0 {
		if tc.shouldStop() {
			return alpha
		}
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
	if move, score, flag, _, found, usable := tt.Probe(p.hash, depth); found {
		hashMove = move
		if pv == nil {
			scoreFromTT := score
			isMateScore := scoreFromTT > Mate-MateScoreGuard || scoreFromTT < -Mate+MateScoreGuard
			if isMateScore {
				if scoreFromTT > 0 {
					scoreFromTT -= ply
				} else {
					scoreFromTT += ply
				}
				// Can use this regardless of depth
				usable = true
			}

			if usable {
				switch flag {
				case ttFlagExact:
					return scoreFromTT
				case ttFlagLower:
					if scoreFromTT >= beta {
						return scoreFromTT
					}
				case ttFlagUpper:
					if scoreFromTT <= alpha {
						return scoreFromTT
					}
				}
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
		if hashMove != 0 && !hashMove.isCapture() {
			eval := p.evaluate()
			// If we are far above beta, we can return soft fail
			if eval >= beta+DeltaMargin*depth {
				return eval
			}
		}
	}

	// Adaptive Null Move Pruning
	if depth >= 3 && !inCheck && !p.isEndgame() {
		R := 3 + depth/6

		undo := p.makeNullMove()
		score := -p.negamax(depth-1-R, -beta, -beta+1, ply+1, nil, tc, ss, 0)
		p.unmakeNullMove(undo)

		if score >= beta {
			return beta
		}
	}

	// ProbCut
	if depth >= 5 && !inCheck && !p.isEndgame() {
		probBeta := beta + 200
		if probBeta <= Mate-MateScoreGuard {
			score := p.negamax(depth-4, probBeta-1, probBeta, ply+1, nil, tc, ss, prevMove)
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

	pvNode := pv != nil
	var pvPtr *[]Move

	var childPVBuf [MaxDepth]Move

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
		if isQuiet {
			quietsTried[quietCount] = m
			quietCount++
		}
		undo := p.makeMove(m)
		childPV := childPVBuf[:0]
		if pvNode {
			pvPtr = &childPV
		}
		var score int

		if p.isDraw() {
			score = contemptValue
		} else {
			// Late move reductions & Principal variation search
			childDepth := depth - 1
			canReduce := childDepth >= LMRMinChildDepth && !inCheck && isQuiet && legalMoves > LMRLateMoveAfter
			var eff int
			if canReduce {
				red := lmrTable[min(depth, MaxDepth)][min(legalMoves, 255)]
				// Reduce worse lines more
				if !pvNode {
					red++
				}
				if history[p.side^1][m.from()][m.to()] < 0 {
					red++
				}
				eff = max(1, childDepth-red)
			}
			if canReduce {
				score = -p.negamax(eff, -alpha-1, -alpha, ply+1, nil, tc, ss, m)
				if score > alpha {
					score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss, m)
				}
			} else {
				if legalMoves > 1 && pvNode {
					score = -p.negamax(childDepth, -alpha-1, -alpha, ply+1, nil, tc, ss, m)
					if score > alpha {
						score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss, m)
					}
				} else {
					score = -p.negamax(childDepth, -beta, -alpha, ply+1, pvPtr, tc, ss, m)
				}
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
			storeScore := score
			if storeScore > Mate-MateScoreGuard {
				storeScore += ply
			} else if storeScore < -Mate+MateScoreGuard {
				storeScore -= ply
			}
			tt.Save(p.hash, m, storeScore, depth, ttFlagLower)
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
			if pv != nil {
				*pv = append(append((*pv)[:0], m), childPV...)
			}
		}
	}

	// Handle draw & checkmate results
	if legalMoves == 0 {
		var result int
		if inCheck {
			result = -Mate + ply
		} else {
			result = 0
		}
		storeScore := result
		if storeScore > Mate-MateScoreGuard {
			storeScore += ply
		} else if storeScore < -Mate+MateScoreGuard {
			storeScore -= ply
		}
		tt.Save(p.hash, Move(0), storeScore, depth, ttFlagExact)
		return result
	}

	flag := ttFlagExact
	if bestScore <= origAlpha {
		flag = ttFlagUpper
	}

	storeScore := bestScore
	if storeScore > Mate-MateScoreGuard {
		storeScore += ply
	} else if storeScore < -Mate+MateScoreGuard {
		storeScore -= ply
	}
	tt.Save(p.hash, bestMove, storeScore, depth, flag)
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

func (p *Position) search(tc *TimeControl) Move {
	var bestMove Move
	var ss [MaxDepth]SearchStack

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
	var pvBuf [MaxDepth]Move
	for depth := 1; depth <= maxDepth; depth++ {
		p.seldepth = depth
		pv := pvBuf[:0]
		var score int

		// Aspiration windows
		if depth >= AspirationStartDepth {
			window := AspirationBase
			low, high := prevScore-window, prevScore+window
			for {
				if low < -Infinity {
					low = -Infinity
				}
				if high > Infinity {
					high = Infinity
				}

				score = p.negamax(depth, low, high, 0, &pv, tc, &ss, 0)

				if tc.shouldStop() {
					break
				}

				if score <= low {
					low -= window
					window *= 2
				} else if score >= high {
					high += window
					window *= 2
				} else {
					break
				}

				if window >= 1000 {
					score = p.negamax(depth, -Infinity, Infinity, 0, &pv, tc, &ss, 0)
					break
				}
			}
		} else {
			score = p.negamax(depth, -Infinity, Infinity, 0, &pv, tc, &ss, 0)
		}
		iterNodes := p.localNodes

		elapsed := time.Since(start)
		elapsedMs := elapsed.Milliseconds()
		nps := int64(0)
		if elapsed > 0 {
			nps = int64(float64(iterNodes) / elapsed.Seconds())
		}

		if tc.shouldStop() {
			break
		}

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
		absScore := abs(score)
		hashfull := tt.Hashfull()
		if absScore >= Mate-MateScoreGuard {
			matePly := Mate - absScore
			mateMoves := (matePly + 1) / 2
			if score > 0 {
				fmt.Printf("info depth %d seldepth %d score mate %d nodes %d time %d nps %d hashfull %d pv",
					depth, p.seldepth, mateMoves, iterNodes, elapsedMs, nps, hashfull)
			} else {
				fmt.Printf("info depth %d seldepth %d score mate -%d nodes %d time %d nps %d hashfull %d pv",
					depth, p.seldepth, mateMoves, iterNodes, elapsedMs, nps, hashfull)
			}
		} else {
			fmt.Printf("info depth %d seldepth %d score cp %d nodes %d time %d nps %d hashfull %d pv",
				depth, p.seldepth, score, iterNodes, elapsedMs, nps, hashfull)
		}
		for _, m := range pv {
			fmt.Printf(" %v", m)
		}
		fmt.Println()

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
	if avail <= 0 {
		tc.optimumMs = minTimeMs
		tc.deadline = time.Now().Add(time.Duration(minTimeMs) * time.Millisecond)
		return
	}

	// Soft target: fair fraction of remaining time plus 3/4 of increment
	optimum := avail/mtg + (i*3)/4
	optimum = max(min(optimum, avail), minTimeMs)

	// Hard emergency ceiling: up to 3.5x soft target, strictly capped at available time
	maxTime := min(optimum*7/2, avail)
	maxTime = max(maxTime, minTimeMs)

	tc.optimumMs = optimum
	tc.deadline = time.Now().Add(time.Duration(maxTime) * time.Millisecond)
}

func (tc *TimeControl) shouldStop() bool {
	return atomic.LoadInt32(&tc.stopped) != 0 || (!tc.deadline.IsZero() && time.Until(tc.deadline) <= 0)
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
	projectedNext := iterTime * 2
	if iterTime > 0 {
		if elapsed+projectedNext > softTarget*3/2 {
			return false
		}
		if projectedNext+continueMargin > remainHard {
			return false
		}
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

	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	count := 0
	for i := 0; i < n; i++ {
		m := movesArr[i]
		if !p.isLegal(m) {
			continue
		}
		undo := p.makeMove(m)
		count += p.perft(depth - 1)
		p.unmakeMove(m, undo)
	}

	return count
}

func (p *Position) perftDivide(depth int) {
	var movesArr [256]Move
	n := p.generateMovesTo(movesArr[:], false)
	total := 0

	for i := 0; i < n; i++ {
		m := movesArr[i]
		if !p.isLegal(m) {
			continue
		}
		undo := p.makeMove(m)
		count := p.perft(depth - 1)
		p.unmakeMove(m, undo)

		fmt.Printf("%v: %d\n", m, count)
		total += count
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

func parseSetOption(parts []string) (name, value string) {
	nameStart, nameEnd, valueStart := -1, -1, -1
	for i, p := range parts {
		if p == "name" && nameStart == -1 {
			nameStart = i + 1
			continue
		}
		if p == "value" && nameStart != -1 && nameEnd == -1 {
			nameEnd = i
			valueStart = i + 1
			break
		}
	}
	if nameStart == -1 {
		return "", ""
	}
	if nameEnd == -1 {
		return strings.Join(parts[nameStart:], " "), ""
	}

	if nameStart >= nameEnd {
		return "", ""
	}
	name = strings.Join(parts[nameStart:nameEnd], " ")
	value = strings.Join(parts[valueStart:], " ")
	return name, value
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
				if cur := currentTC.Swap(nil); cur != nil {
					cur.Stop()
				}
				searchWG.Wait()
				InitTT(sizeMB)
				fmt.Printf("info string Hash set to %d MB\n", sizeMB)
			} else {
				fmt.Printf("info string setoption %q = %q (ignored)\n", name, value)
			}

		case "ucinewgame":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			searchWG.Wait()
			tt.Clear()
			clearHeuristics()
			pos.setStartPos()

		case "position":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			searchWG.Wait()
			if len(parts) < 2 {
				fmt.Println("# Error: position requires arguments")
				continue
			}
			moveIdx := -1
			for i := 2; i < len(parts); i++ {
				if parts[i] == "moves" {
					moveIdx = i
					break
				}
			}

			if parts[1] == "startpos" {
				pos.setStartPos()
			} else if parts[1] == "fen" {
				fenParts := []string{}
				for i := 2; i < len(parts); i++ {
					if parts[i] == "moves" {
						moveIdx = i
						break
					}
					fenParts = append(fenParts, parts[i])
				}
				pos.setFEN(strings.Join(fenParts, " "))
			}

			if moveIdx != -1 && moveIdx+1 < len(parts) {
				for _, mvStr := range parts[moveIdx+1:] {
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
			}
		case "go":
			if cur := currentTC.Swap(nil); cur != nil {
				cur.Stop()
			}
			searchWG.Wait()

			tc := &TimeControl{}

			for i := 1; i < len(parts); i++ {
				switch parts[i] {
				case "wtime":
					if i+1 < len(parts) {
						tc.wtime, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "btime":
					if i+1 < len(parts) {
						tc.btime, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "winc":
					if i+1 < len(parts) {
						tc.winc, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "binc":
					if i+1 < len(parts) {
						tc.binc, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "movestogo":
					if i+1 < len(parts) {
						tc.movestogo, _ = strconv.Atoi(parts[i+1])
						i++
					}
				case "depth":
					if i+1 < len(parts) {
						tc.depth, _ = strconv.Atoi(parts[i+1])
						i++
					}
				case "movetime":
					if i+1 < len(parts) {
						tc.movetime, _ = strconv.ParseInt(parts[i+1], 10, 64)
						i++
					}
				case "infinite":
					tc.infinite = true
				}
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
					sq := r*8 + f
					c, pt, ok := pos.pieceAt(sq)
					if ok {
						piece := "PNBRQK"[pt]
						if c == Black {
							piece += 32
						}
						fmt.Printf(" %c", piece)
					} else {
						fmt.Print(" .")
					}
				}
				fmt.Printf(" |%d\n", r+1)
			}
			fmt.Println("  ----------------")
			fmt.Println("   a b c d e f g h")
			fmt.Printf("Side to move: %s\n", map[int]string{White: "White", Black: "Black"}[pos.side])
			fmt.Printf("Hash: %x\n\n", pos.hash)

		case "eval":
			score := pos.evaluate()
			fmt.Printf("Evaluation: %+d (from %s's perspective)\n", score, map[int]string{White: "White", Black: "Black"}[pos.side])

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
			for sq := 0; sq < 64; sq++ {
				c, pt, ok := pos.pieceAt(sq)
				if ok {
					expectedHash ^= zobristPiece[c][pt][sq]
					if pt == Pawn {
						expectedPawnHash ^= zobristPiece[c][Pawn][sq]
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
					timeStr := ""
					if elapsed < time.Second {
						timeStr = fmt.Sprintf("%d ms", elapsed.Milliseconds())
					} else {
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
// set GOAMD64=v3 && go build -trimpath -ldflags "-s -w" -gcflags "all=-B" -o Soomi-V1.2.0B.exe soomi.go
