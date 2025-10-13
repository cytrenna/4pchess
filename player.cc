// Multi-word bitboard for a 16x16 embedding (256 bits -> 4 * 64)
#include <cstdint>
#include <array>
#include <cassert>

struct BB256 {
    std::array<uint64_t,4> w; // w[0] = bits 0..63 (least significant), w[3] = 192..255

    static BB256 zero() { return {{0,0,0,0}}; }
    static BB256 all()  { return {{~0ULL,~0ULL,~0ULL,~0ULL}}; }

    bool any() const { return (w[0]|w[1]|w[2]|w[3]) != 0ULL; }
    void reset() { w = {{0,0,0,0}}; }

    // bit ops
    BB256& operator|=(const BB256& b){ for(int i=0;i<4;++i) w[i]|=b.w[i]; return *this; }
    BB256& operator&=(const BB256& b){ for(int i=0;i<4;++i) w[i]&=b.w[i]; return *this; }
    BB256 operator~() const { return {{~w[0],~w[1],~w[2],~w[3]}}; }
    BB256 operator|(const BB256& b) const { BB256 r=*this; r|=b; return r; }
    BB256 operator&(const BB256& b) const { BB256 r=*this; r&=b; return r; }

    // set/test bit by index 0..255
    void set_bit(unsigned idx){ assert(idx < 256); w[idx>>6] |= (1ULL << (idx & 63)); }
    bool test_bit(unsigned idx) const { return (w[idx>>6] >> (idx & 63)) & 1ULL; }
    void clear_bit(unsigned idx){ w[idx>>6] &= ~(1ULL << (idx & 63)); }

    // popcount and lsb/msb:
    int popcount() const {
        return __builtin_popcountll(w[0]) + __builtin_popcountll(w[1])
             + __builtin_popcountll(w[2]) + __builtin_popcountll(w[3]);
    }
    // return index of least-significant 1 bit, or -1 if empty
    int lsb_index() const {
        for(int i=0;i<4;++i) if (w[i]) return (i<<6) + __builtin_ctzll(w[i]);
        return -1;
    }
    // pop and return lsb index (clears it). returns -1 if empty
    int pop_lsb() {
        for(int i=0;i<4;++i) {
            if (w[i]) {
                uint64_t low = w[i] & -w[i];
                int bit = __builtin_ctzll(w[i]);
                w[i] &= w[i] - 1; // clear lsb
                return (i<<6) + bit;
            }
        }
        return -1;
    }
};

// SHIFT helpers: generic shift left/right by any small s (0..255).
inline BB256 shl(const BB256& b, unsigned s) {
    if (s == 0) return b;
    BB256 r = BB256::zero();
    unsigned wshift = s >> 6;
    unsigned bshift = s & 63;
    for (int i = 3; i >= 0; --i) {
        int src = i - (int)wshift;
        if (src < 0) continue;
        uint64_t lo = b.w[src] << bshift;
        uint64_t hi = 0;
        if (bshift && src-1 >= 0) hi = b.w[src-1] >> (64 - bshift);
        r.w[i] = lo | hi;
    }
    return r;
}
inline BB256 shr(const BB256& b, unsigned s) {
    if (s == 0) return b;
    BB256 r = BB256::zero();
    unsigned wshift = s >> 6;
    unsigned bshift = s & 63;
    for (int i = 0; i < 4; ++i) {
        int src = i + (int)wshift;
        if (src > 3) continue;
        uint64_t lo = b.w[src] >> bshift;
        uint64_t hi = 0;
        if (bshift && src+1 <= 3) hi = b.w[src+1] << (64 - bshift);
        r.w[i] = lo | hi;
    }
    return r;
}

// Precomputed file masks for 16 files (file 0 = A .. file 15 = P)
BB256 file_mask(unsigned file) {
    assert(file < 16);
    BB256 m = BB256::zero();
    for(unsigned r=0;r<16;++r) m.set_bit(r*16 + file);
    return m;
}

// We'll create a handful of masks once:
struct Masks {
    BB256 VALID;        // caller must set: playable squares (e.g., 14x14 with cut corners)
    BB256 FILE_A, FILE_P, FILE_O, FILE_N; // leftmost and rightmost file masks and the two-right-most
    // convenience negations:
    BB256 not_FILE_A, not_FILE_P, not_FILE_OP; // OP = O or P (two rightmost)
} masks;

// Directional single-step safe shifts (use masks to prevent wrap)
// Assumes masks.FILE_A / FILE_P are filled for 16x16 layout
inline BB256 shift_north(const BB256 &b) { return shl(b, 16) & masks.VALID; }
inline BB256 shift_south(const BB256 &b) { return shr(b, 16) & masks.VALID; }
inline BB256 shift_east(const BB256 &b)  { return shl( b & masks.not_FILE_P, 1) & masks.VALID; }
inline BB256 shift_west(const BB256 &b)  { return shr( b & masks.not_FILE_A, 1) & masks.VALID; }
inline BB256 shift_ne(const BB256 &b)    { return shl( b & masks.not_FILE_P, 17) & masks.VALID; } // +16 +1
inline BB256 shift_nw(const BB256 &b)    { return shl( b & masks.not_FILE_A, 15) & masks.VALID; } // +16 -1
inline BB256 shift_se(const BB256 &b)    { return shr( b & masks.not_FILE_P, 15) & masks.VALID; } // -16 +1 => -15
inline BB256 shift_sw(const BB256 &b)    { return shr( b & masks.not_FILE_A, 17) & masks.VALID; } // -16 -1 => -17

// Slider rays generator from a single square, direction function passed as a lambda
template<typename ShiftFunc>
BB256 ray_attacks_from(unsigned sq_idx, const BB256 &occupancy, ShiftFunc shift) {
    BB256 mask = BB256::zero();
    BB256 cur = BB256::zero();
    cur.set_bit(sq_idx);
    while (true) {
        cur = shift(cur);
        if (!cur.any()) break;
        mask |= cur;
        // stop if blocking piece present on this step
        BB256 inter = mask & occupancy;
        if (inter.any()) break; // there is a blocker in the ray we just added
        // continue
    }
    return mask;
}

// High-level helper: rook and bishop attacks for a square given current occupancy
BB256 rook_attacks(unsigned sq, const BB256 &occ) {
    BB256 r = BB256::zero();
    r |= ray_attacks_from(sq, occ, shift_north);
    r |= ray_attacks_from(sq, occ, shift_south);
    r |= ray_attacks_from(sq, occ, shift_east);
    r |= ray_attacks_from(sq, occ, shift_west);
    return r;
}
BB256 bishop_attacks(unsigned sq, const BB256 &occ) {
    BB256 r = BB256::zero();
    r |= ray_attacks_from(sq, occ, shift_ne);
    r |= ray_attacks_from(sq, occ, shift_nw);
    r |= ray_attacks_from(sq, occ, shift_se);
    r |= ray_attacks_from(sq, occ, shift_sw);
    return r;
}
BB256 queen_attacks(unsigned sq, const BB256 &occ) {
    BB256 r = rook_attacks(sq, occ);
    r |= bishop_attacks(sq, occ);
    return r;
}

// King attacks = one-step in 8 directions from bitboard 'b' (or for single square)
BB256 king_attacks_from_bb(const BB256 &b) {
    BB256 r = BB256::zero();
    r |= shift_north(b);
    r |= shift_south(b);
    r |= shift_east(b);
    r |= shift_west(b);
    r |= shift_ne(b);
    r |= shift_nw(b);
    r |= shift_se(b);
    r |= shift_sw(b);
    return r;
}
BB256 king_attacks(unsigned sq) { BB256 tmp = BB256::zero(); tmp.set_bit(sq); return king_attacks_from_bb(tmp); }

// Knight: offsets encoded with required source exclusion masks to avoid wrap.
// On 16x16 these deltas are: ±33, ±31, ±18, ±14, ±33 means r+2 f+1 (32+1)
struct KnightOffset { int delta; const BB256* source_mask; }; // mask of allowed source bits (i.e., bits to keep before shifting)
BB256 knight_attacks(unsigned sq) {
    BB256 res = BB256::zero();
    BB256 single = BB256::zero(); single.set_bit(sq);

    // Offsets and which source files must be excluded.
    // For file +1: exclude FILE_P. For +2: exclude FILE_O and FILE_P. For -1: exclude FILE_A. etc.
    // We'll inline the 8 offsets with their masks:
    // +33 (r+2, f+1) : shl(src & not_FILE_P, 33)
    // +31 (r+2, f-1) : shl(src & not_FILE_A, 31)
    // +18 (r+1, f+2) : shl(src & not_FILE_OP, 18)
    // +14 (r+1, f-2) : shl(src & not_FILE_AO??) -> we'll compute via two-file-left mask
    // For simplicity, we precompute masks earlier in 'masks' (not_FILE_P, not_FILE_A, not_FILE_OP).

    // +33
    res |= shl(single & masks.not_FILE_P, 33);
    // +31
    res |= shl(single & masks.not_FILE_A, 31);
    // +18 (file +2): exclude files O and P (two rightmost)
    res |= shl(single & masks.not_FILE_OP, 18);
    // +14 (file -2): exclude files A and B -> prepare not_FILE_AB if needed
    // We didn't precompute all, so build inline:
    BB256 not_FILE_AB = (~file_mask(0)) & (~file_mask(1));
    res |= shl(single & not_FILE_AB, 14);

    // negative deltas:
    res |= shr(single & masks.not_FILE_P, 31); // -31  (down2 right1 -> -32 +1 = -31)
    res |= shr(single & masks.not_FILE_A, 33); // -33
    res |= shr(single & masks.not_FILE_OP, 14); // -14 (down1 right2 => -16 +2 = -14)
    res |= shr(single & not_FILE_AB, 18); // -18 (down1 left2 => -16 -2 = -18)

    return res & masks.VALID;
}