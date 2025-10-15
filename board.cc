// fourpc.cpp  —  4-Player Teams chess core + perft (C++17)
// Layout: 14x14 with 3x3 cutouts (Chess.com Teams). Pawns push toward center.
//
// Features:
// • 3×64b bitboards (covers 160 squares)
// • Geometry (16×16 mailbox), attacks & legal move gen
// • En passant (double push, EP target, lifetime across two opponent turns)
// • Castling (all four colors; path empty, not-in-check, pass squares safe)
// • Make/undo with reversible state
// • Start position + perft / divide
//
// Build demo: g++ -O3 -std=c++17 fourpc.cpp -o fourpc -DFPC_PERFT_MAIN

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace FPC {

// ---------------- Bitboard ----------------

struct BB {
    std::array<uint64_t,3> w{};

    inline void clear(){ w = {0,0,0}; }
    inline bool any() const { return w[0] | w[1] | w[2]; }
    inline bool none() const { return !any(); }

    static inline BB single(int sq){
        BB b;
        if (sq < 64) b.w[0] = 1ULL << sq;
        else if (sq < 128) b.w[1] = 1ULL << (sq-64);
        else b.w[2] = 1ULL << (sq-128);
        return b;
    }
    inline void set(int sq){
        if (sq < 64) w[0] |= (1ULL << sq);
        else if (sq < 128) w[1] |= (1ULL << (sq-64));
        else w[2] |= (1ULL << (sq-128));
    }
    inline void reset(int sq){
        if (sq < 64) w[0] &= ~(1ULL << sq);
        else if (sq < 128) w[1] &= ~(1ULL << (sq-64));
        else w[2] &= ~(1ULL << (sq-128));
    }
    inline bool test(int sq) const {
        if (sq < 64) return (w[0] >> sq) & 1ULL;
        else if (sq < 128) return (w[1] >> (sq-64)) & 1ULL;
        else return (w[2] >> (sq-128)) & 1ULL;
    }

    static inline int popcount64(uint64_t x){ return __builtin_popcountll(x); }
    inline int popcount() const { return popcount64(w[0]) + popcount64(w[1]) + popcount64(w[2]); }

    inline int pop_lsb(){
        for (int i=0;i<3;i++){
            if (w[i]){
                uint64_t x = w[i] & -w[i];
                int bit = __builtin_ctzll(w[i]);
                w[i] ^= x;
                return bit + 64*i;
            }
        }
        return -1;
    }

    friend inline BB operator|(BB a, const BB& b){ a.w[0]|=b.w[0]; a.w[1]|=b.w[1]; a.w[2]|=b.w[2]; return a; }
    friend inline BB operator&(BB a, const BB& b){ a.w[0]&=b.w[0]; a.w[1]&=b.w[1]; a.w[2]&=b.w[2]; return a; }
    friend inline BB operator^(BB a, const BB& b){ a.w[0]^=b.w[0]; a.w[1]^=b.w[1]; a.w[2]^=b.w[2]; return a; }
};

// ---------------- Enums/Constants ----------------

enum Color : int { RED=0, BLUE=1, YELLOW=2, GREEN=3, COLOR_N=4 };
enum Piece : int { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, PIECE_N=6 };
enum Team  : int { TEAM_RY=0, TEAM_BG=1 };

static constexpr Team PLAYER_TEAM[COLOR_N] = {
    TEAM_RY, TEAM_BG, TEAM_RY, TEAM_BG
};

constexpr int BOARD_SIDE = 14;
constexpr int SQUARE_COUNT = 160;
constexpr int MAILBOX_SIDE = 16;
constexpr int INVALID = -1;

constexpr int DIR_N = +16, DIR_S = -16, DIR_E = +1, DIR_W = -1;
static const int KING_DIRS[8]   = {+16,-16,+1,-1,+17,+15,-15,-17};
static const int KNIGHT_DIRS[8] = {+33,+31,+18,-14,-33,-31,-18,+14};

// Playable squares: 14×14 minus 3×3 in each corner (Chess.com layout)
inline bool isPlayableRF(int r,int f){
    if (r<0||r>=14||f<0||f>=14) return false;
    bool tl=(r>=11&&f<=2), tr=(r>=11&&f>=11);
    bool bl=(r<=2&&f<=2),  br=(r<=2&&f>=11);
    return !(tl||tr||bl||br);
}

static int MB_TO_SQ[256];
static int SQ_TO_MB[SQUARE_COUNT];
inline int mIndex(int r,int f){ return (r<<4)|f; }
inline int mbRank(int mb){ return mb >> 4; }
inline int mbFile(int mb){ return mb & 15; }

static void init_geometry(){
    static bool done=false; if (done) return; done=true;
    std::fill(std::begin(MB_TO_SQ), std::end(MB_TO_SQ), INVALID);
    int sq=0;
    for (int r=0;r<14;r++)
        for (int f=0;f<14;f++)
            if (isPlayableRF(r,f)){
                int mb=mIndex(r,f);
                MB_TO_SQ[mb]=sq;
                SQ_TO_MB[sq]=mb;
                sq++;
            }
    assert(sq==SQUARE_COUNT);
}

inline int sqRF(int r,int f){ return MB_TO_SQ[mIndex(r,f)]; }

// ---------------- Pawn directions (toward center) ----------------

struct PawnDirs { int push, capL, capR; };
static PawnDirs PAWN[COLOR_N] = {
    { DIR_N, DIR_N-1, DIR_N+1 },   // RED pushes north
    { DIR_E, DIR_E-16, DIR_E+16 }, // BLUE pushes east
    { DIR_S, DIR_S-1, DIR_S+1 },   // YELLOW pushes south
    { DIR_W, DIR_W-16, DIR_W+16 }, // GREEN pushes west
};

// ---------------- Move (flags: EP, DoublePush, Castle) ----------------

struct Move {
    uint32_t v{};
    enum : uint32_t {
        EP_FLAG = 1u<<28,
        DP_FLAG = 1u<<29,
        CS_FLAG = 1u<<30
    };
    Move()=default;
    Move(int from,int to,int piece,int capPiece=0,int capColor=0,uint32_t flags=0){
        v = (from) | (to<<8) | (piece<<16) | (capPiece<<20) | (capColor<<24) | flags;
    }
    inline int  from()     const { return  v       & 0xFF; }
    inline int  to()       const { return (v>>8)  & 0xFF; }
    inline int  piece()    const { return (v>>16) & 0xF;  }
    inline int  capPiece() const { return (v>>20) & 0xF;  }
    inline int  capColor() const { return (v>>24) & 0xF;  }
    inline bool isEP()     const { return (v & EP_FLAG)!=0; }
    inline bool isDP()     const { return (v & DP_FLAG)!=0; }
    inline bool isCastle() const { return (v & CS_FLAG)!=0; }
};

// ---------------- Castling rights & geometry ----------------

inline int CR_K(Color c){ return 1 << (2*int(c) + 0); } // short
inline int CR_Q(Color c){ return 1 << (2*int(c) + 1); } // long

struct CastleGeom {
    int kFrom, rFrom, kTo, rTo;
    int empty[3]; int emptyN;      // must be empty
    int pass[2];  int passN;       // king transit squares: not attacked
};

static CastleGeom C[COLOR_N][2];

static void init_castling(){
    // RED bottom: king (0,7), rooks (0,3) & (0,10)
    C[RED][0] = { sqRF(0,7),  sqRF(0,10), sqRF(0,9),  sqRF(0,8),  { sqRF(0,8), sqRF(0,9), -1 }, 2,
                  { sqRF(0,8), sqRF(0,9) }, 2 };
    C[RED][1] = { sqRF(0,7),  sqRF(0,3),  sqRF(0,5),  sqRF(0,6),  { sqRF(0,6), sqRF(0,5), sqRF(0,4) }, 3,
                  { sqRF(0,6), sqRF(0,5) }, 2 };

    // YELLOW top
    C[YELLOW][0] = { sqRF(13,7), sqRF(13,10), sqRF(13,9), sqRF(13,8), { sqRF(13,8), sqRF(13,9), -1 }, 2,
                     { sqRF(13,8), sqRF(13,9) }, 2 };
    C[YELLOW][1] = { sqRF(13,7), sqRF(13,3),  sqRF(13,5), sqRF(13,6), { sqRF(13,6), sqRF(13,5), sqRF(13,4) }, 3,
                     { sqRF(13,6), sqRF(13,5) }, 2 };

    // BLUE left: king (7,0), rooks (3,0) & (10,0)
    C[BLUE][0] = { sqRF(7,0),  sqRF(10,0), sqRF(9,0),  sqRF(8,0),  { sqRF(8,0), sqRF(9,0), -1 }, 2,
                   { sqRF(8,0), sqRF(9,0) }, 2 };
    C[BLUE][1] = { sqRF(7,0),  sqRF(3,0),  sqRF(5,0),  sqRF(6,0),  { sqRF(6,0), sqRF(5,0), sqRF(4,0) }, 3,
                   { sqRF(6,0), sqRF(5,0) }, 2 };

    // GREEN right: king (7,13), rooks (10,13) & (3,13)
    C[GREEN][0] = { sqRF(7,13), sqRF(10,13), sqRF(9,13), sqRF(8,13), { sqRF(8,13), sqRF(9,13), -1 }, 2,
                    { sqRF(8,13), sqRF(9,13) }, 2 };
    C[GREEN][1] = { sqRF(7,13), sqRF(3,13),  sqRF(5,13), sqRF(6,13), { sqRF(6,13), sqRF(5,13), sqRF(4,13) }, 3,
                    { sqRF(6,13), sqRF(5,13) }, 2 };
}

// ---------------- Undo & Board ----------------

struct Undo {
    Move move;
    int capturedPiece{-1};
    int capturedColor{-1};
    int kingSqPrev[COLOR_N];
    int stmPrev;

    // EP snapshot
    int epSquarePrev{-1}, epMaskPrev{0}, epRemainPrev{0};

    // Castling rights snapshot
    int castleRightsPrev{0};
};

struct Board {
    BB bb[COLOR_N][PIECE_N]{};
    BB occColor[COLOR_N]{};
    BB occAll{};
    Color sideToMove{RED};
    int kingSq[COLOR_N]{-1,-1,-1,-1};

    // En Passant state
    int epSquare{-1};   // target square to move TO on an EP capture
    int epMask{0};      // which colors may EP-capture (both opponents)
    int epRemain{0};    // remaining opponent turns (2 -> 1 -> 0)

    // Castling rights
    int castleRights{0};  // bitmask using CR_K/CR_Q

    // convenience
    inline int colorBit(Color c) const { return 1<<int(c); }
    inline Team teamOf(Color c) const { return PLAYER_TEAM[c]; }
    inline bool hasK(Color c) const { return (castleRights & CR_K(c)) != 0; }
    inline bool hasQ(Color c) const { return (castleRights & CR_Q(c)) != 0; }
    inline void clearK(Color c){ castleRights &= ~CR_K(c); }
    inline void clearQ(Color c){ castleRights &= ~CR_Q(c); }

    inline void rebuild(){
        occAll.clear();
        for(int c=0;c<COLOR_N;c++){
            occColor[c].clear();
            for(int p=0;p<PIECE_N;p++) occColor[c]|=bb[c][p];
            occAll|=occColor[c];
        }
        for(int c=0;c<COLOR_N;c++){
            int k=-1; BB t=bb[c][KING];
            if (t.any()) k=t.pop_lsb();
            kingSq[c]=k;
        }
    }
    inline void put(Color c, Piece p, int sq){
        bb[c][p].set(sq); occColor[c].set(sq); occAll.set(sq);
        if (p==KING) kingSq[c]=sq;
    }
    inline void remove(Color c, Piece p, int sq){
        bb[c][p].reset(sq); occColor[c].reset(sq); occAll.reset(sq);
        if (p==KING) kingSq[c]=-1;
    }
    inline bool occupied(int sq) const { return occAll.test(sq); }
    inline bool friendly(Color c,int sq) const { return occColor[c].test(sq); }

    inline Piece pieceAt(Color c,int sq) const {
        for(int p=0;p<PIECE_N;p++) if (bb[c][p].test(sq)) return (Piece)p;
        return (Piece)PIECE_N;
    }
    inline Color colorAt(int sq) const {
        for(int c=0;c<COLOR_N;c++) if (occColor[c].test(sq)) return (Color)c;
        return (Color)COLOR_N;
    }

    // --- attack helpers ---

    inline BB attacks_from_pawns(Color c, const BB& pawns) const {
        BB a; BB t=pawns;
        while (t.any()){
            int sq = t.pop_lsb();
            int mb = SQ_TO_MB[sq];
            for (int d : {PAWN[c].capL, PAWN[c].capR}){
                int tsq = MB_TO_SQ[mb + d];
                if (tsq!=INVALID) a.set(tsq);
            }
        }
        return a;
    }
    inline BB attacks_from_knights(const BB& knights) const {
        BB a; BB t=knights;
        while (t.any()){
            int sq=t.pop_lsb();
            int mb=SQ_TO_MB[sq];
            for (int d:KNIGHT_DIRS){
                int tsq=MB_TO_SQ[mb+d];
                if (tsq!=INVALID) a.set(tsq);
            }
        }
        return a;
    }
    inline BB attacks_from_kings(const BB& kings) const {
        BB a; BB t=kings;
        while (t.any()){
            int sq=t.pop_lsb();
            int mb=SQ_TO_MB[sq];
            for (int d:KING_DIRS){
                int tsq=MB_TO_SQ[mb+d];
                if (tsq!=INVALID) a.set(tsq);
            }
        }
        return a;
    }
    inline BB attacks_from_sliders_dir(const BB& sliders, const int *dirIdx, int dirCount) const {
        BB a; BB t=sliders;
        while (t.any()){
            int from=t.pop_lsb();
            int mb=SQ_TO_MB[from];
            for (int k=0;k<dirCount;k++){
                int d=dirIdx[k], tmb=mb+d;
                while (true){
                    int tsq=MB_TO_SQ[tmb];
                    if (tsq==INVALID) break;
                    a.set(tsq);
                    if (occupied(tsq)) break;
                    tmb+=d;
                }
            }
        }
        return a;
    }
    inline BB attacks_from_bishops(const BB& bbs) const {
        static const int DIRS[4]={+17,+15,-15,-17};
        return attacks_from_sliders_dir(bbs, DIRS, 4);
    }
    inline BB attacks_from_rooks(const BB& rrs) const {
        static const int DIRS[4]={+16,-16,+1,-1};
        return attacks_from_sliders_dir(rrs, DIRS, 4);
    }
    inline BB attacks_from_queens(const BB& qq) const {
        return attacks_from_bishops(qq) | attacks_from_rooks(qq);
    }

    inline bool square_attacked_by_opponents(int sq, Color me) const {
        Team myTeam = teamOf(me);
        int mb = SQ_TO_MB[sq];

        // Knights
        for (int d:KNIGHT_DIRS){
            int fsq = MB_TO_SQ[mb - d];
            if (fsq!=INVALID){
                for (int c=0;c<COLOR_N;c++){
                    if (teamOf((Color)c)==myTeam) continue;
                    if (bb[c][KNIGHT].test(fsq)) return true;
                }
            }
        }
        // Kings
        for (int d:KING_DIRS){
            int fsq = MB_TO_SQ[mb - d];
            if (fsq!=INVALID){
                for (int c=0;c<COLOR_N;c++){
                    if (teamOf((Color)c)==myTeam) continue;
                    if (bb[c][KING].test(fsq)) return true;
                }
            }
        }
        // Pawns (reverse captures)
        for (int c=0;c<COLOR_N;c++){
            if (teamOf((Color)c)==myTeam) continue;
            for (int d : {PAWN[c].capL, PAWN[c].capR}){
                int fsq = MB_TO_SQ[mb - d];
                if (fsq!=INVALID && bb[c][PAWN].test(fsq)) return true;
            }
        }
        // Sliders: scan rays outward from sq
        auto ray_hit = [&](int d, Piece need1, Piece need2){
            int tmb = mb + d;
            while (true){
                int tsq = MB_TO_SQ[tmb];
                if (tsq==INVALID) return false;
                if (occupied(tsq)){
                    Color oc = colorAt(tsq);
                    if (oc!=COLOR_N && teamOf(oc)!=myTeam){
                        Piece p = pieceAt(oc, tsq);
                        if (p==need1 || p==need2) return true;
                    }
                    return false;
                }
                tmb += d;
            }
        };
        if (ray_hit(+1,ROOK,QUEEN)  || ray_hit(-1,ROOK,QUEEN) ||
            ray_hit(+16,ROOK,QUEEN) || ray_hit(-16,ROOK,QUEEN)||
            ray_hit(+17,BISHOP,QUEEN)||ray_hit(+15,BISHOP,QUEEN)||
            ray_hit(-15,BISHOP,QUEEN)||ray_hit(-17,BISHOP,QUEEN))
            return true;

        return false;
    }

    inline bool in_check(Color c) const {
        int k = kingSq[c];
        return (k!=-1) && square_attacked_by_opponents(k, c);
    }

    // --- helpers for doubles/EP ---
    inline bool pawn_on_home(Color c, int fromSq){
        int mb = SQ_TO_MB[fromSq];
        int r = mbRank(mb), f = mbFile(mb);
        switch (c){
            case RED:    return r == 1;
            case YELLOW: return r == 12;
            case BLUE:   return f == 1;
            case GREEN:  return f == 12;
            default: return false;
        }
    }
    inline bool double_push_targets(Color c, int fromSq, int& step1Sq, int& step2Sq){
        int mb = SQ_TO_MB[fromSq];
        int s1 = MB_TO_SQ[mb + PAWN[c].push];
        if (s1 == INVALID) return false;
        int s2 = MB_TO_SQ[mb + 2*PAWN[c].push];
        if (s2 == INVALID) return false;
        step1Sq = s1; step2Sq = s2;
        return true;
    }

    // --- castling rights updates ---
    inline void revoke_rights_on_move(Color us, Piece p, int fromSq){
        if (p==KING){ clearK(us); clearQ(us); return; }
        if (p==ROOK){
            const CastleGeom& K = C[us][0];
            const CastleGeom& Q = C[us][1];
            if (fromSq == K.rFrom) clearK(us);
            if (fromSq == Q.rFrom) clearQ(us);
        }
    }
    inline void revoke_rights_on_capture(Color victimColor, int atSq){
        const CastleGeom& K = C[victimColor][0];
        const CastleGeom& Q = C[victimColor][1];
        if (atSq == K.rFrom) clearK(victimColor);
        if (atSq == Q.rFrom) clearQ(victimColor);
    }

    inline bool canCastleSide(Color us, int side /*0=short,1=long*/) const {
        const CastleGeom& G = C[us][side];

        if (side==0 && !hasK(us)) return false;
        if (side==1 && !hasQ(us)) return false;
        if (!bb[us][KING].test(G.kFrom)) return false;
        if (!bb[us][ROOK].test(G.rFrom)) return false;

        for (int i=0;i<G.emptyN;i++){
            int s = G.empty[i];
            if (s==INVALID) continue;
            if (occupied(s)) return false;
        }
        if (in_check(us)) return false;
        for (int i=0;i<G.passN;i++){
            if (square_attacked_by_opponents(G.pass[i], us)) return false;
        }
        return true;
    }

    // -------- Pseudo move gen --------
    void generatePseudo(std::vector<Move>& mv) const {
        const Color us = sideToMove;

        for (int p=0; p<PIECE_N; ++p) {
            BB pcs = bb[us][p];
            while (pcs.any()) {
                int from = pcs.pop_lsb();
                int mb = SQ_TO_MB[from];

                if (p==PAWN){
                    // single push
                    int to = MB_TO_SQ[mb + PAWN[us].push];
                    if (to!=INVALID && !occupied(to))
                        mv.emplace_back(from,to,p);

                    // double push
                    if (pawn_on_home(us, from)){
                        int s1, s2;
                        if (double_push_targets(us, from, s1, s2) && !occupied(s1) && !occupied(s2)){
                            mv.emplace_back(from, s2, p, 0, 0, Move::DP_FLAG);
                        }
                    }

                    // normal captures
                    for (int d : {PAWN[us].capL, PAWN[us].capR}){
                        int tsq = MB_TO_SQ[mb + d];
                        if (tsq==INVALID || occColor[us].test(tsq)) continue;
                        Color oc = colorAt(tsq);
                        if (oc!=COLOR_N && PLAYER_TEAM[oc]!=PLAYER_TEAM[us])
                            mv.emplace_back(from,tsq,p,pieceAt(oc,tsq),oc);
                    }

                    // en passant captures
                    if (epSquare != -1 && epRemain > 0 && (epMask & colorBit(us))){
                        for (int d : {PAWN[us].capL, PAWN[us].capR}){
                            int fromCand = MB_TO_SQ[SQ_TO_MB[epSquare] - d];
                            if (fromCand == INVALID) continue;
                            if (!bb[us][PAWN].test(fromCand)) continue;

                            int capturedSq = MB_TO_SQ[SQ_TO_MB[epSquare] - PAWN[us].push];
                            if (capturedSq == INVALID) continue;
                            Color oc = colorAt(capturedSq);
                            if (oc==COLOR_N || PLAYER_TEAM[oc]==PLAYER_TEAM[us]) continue;
                            if (!bb[oc][PAWN].test(capturedSq)) continue;

                            mv.emplace_back(fromCand, epSquare, p, PAWN, oc, Move::EP_FLAG);
                        }
                    }
                }
                else if (p==KNIGHT){
                    for (int d:KNIGHT_DIRS){
                        int tsq=MB_TO_SQ[mb+d];
                        if (tsq==INVALID || occColor[us].test(tsq)) continue;
                        Color oc=colorAt(tsq);
                        if (oc==COLOR_N || PLAYER_TEAM[oc]!=PLAYER_TEAM[us])
                            mv.emplace_back(from,tsq,p,(oc==COLOR_N?0:pieceAt(oc,tsq)),oc);
                    }
                }
                else if (p==BISHOP || p==ROOK || p==QUEEN){
                    static const int DELTAS[8]={+16,-16,+1,-1,+17,+15,-15,-17};
                    int s=(p==BISHOP?4:(p==ROOK?0:0));
                    int e=(p==BISHOP?8:(p==ROOK?4:8));
                    for (int i=s;i<e;i++){
                        int d=DELTAS[i], tmb=mb+d;
                        while (true){
                            int tsq=MB_TO_SQ[tmb];
                            if (tsq==INVALID) break;
                            if (occColor[us].test(tsq)) break;
                            Color oc=colorAt(tsq);
                            if (oc==COLOR_N) mv.emplace_back(from,tsq,p);
                            else {
                                if (PLAYER_TEAM[oc]!=PLAYER_TEAM[us])
                                    mv.emplace_back(from,tsq,p,pieceAt(oc,tsq),oc);
                                break;
                            }
                            tmb+=d;
                        }
                    }
                }
                else if (p==KING){
                    for (int d:KING_DIRS){
                        int tsq=MB_TO_SQ[mb+d];
                        if (tsq==INVALID || occColor[us].test(tsq)) continue;
                        Color oc=colorAt(tsq);
                        if (oc==COLOR_N || PLAYER_TEAM[oc]!=PLAYER_TEAM[us])
                            mv.emplace_back(from,tsq,p,(oc==COLOR_N?0:pieceAt(oc,tsq)),oc);
                    }
                    // Castling
                    if (canCastleSide(us, 0)) {
                        const CastleGeom& G = C[us][0];
                        mv.emplace_back(G.kFrom, G.kTo, KING, 0, 0, Move::CS_FLAG);
                    }
                    if (canCastleSide(us, 1)) {
                        const CastleGeom& G = C[us][1];
                        mv.emplace_back(G.kFrom, G.kTo, KING, 0, 0, Move::CS_FLAG);
                    }
                }
            }
        }
    }

    // -------- Legal move gen (filter self-check) --------
    void generateLegal(std::vector<Move>& legal) {
        std::vector<Move> pseudo;
        pseudo.reserve(256);
        generatePseudo(pseudo);

        Undo u;
        for (const auto& m : pseudo){
            makeMove(m,u);
            bool ok = !in_check((Color)u.stmPrev);
            undoMove(u);
            if (ok) legal.push_back(m);
        }
    }

    // -------- Make / Undo --------
    void makeMove(const Move& m, Undo& u){
        u.move=m;
        std::memcpy(u.kingSqPrev, kingSq, sizeof(kingSq));
        u.stmPrev = sideToMove;

        // Save EP and rights
        u.epSquarePrev = epSquare;
        u.epMaskPrev   = epMask;
        u.epRemainPrev = epRemain;
        u.castleRightsPrev = castleRights;

        const int from=m.from(), to=m.to();
        const Color us=sideToMove;
        const Piece p=(Piece)m.piece();

        // EP lifetime decays for opponents; we handle after knowing move type
        auto advance_turn = [&]{ sideToMove = (Color)((sideToMove + 1) % 4); };

        // rights on our piece move
        revoke_rights_on_move(us, p, from);

        // --- EP move ---
        if (m.isEP()){
            int capSq = MB_TO_SQ[SQ_TO_MB[to] - PAWN[us].push];
            Color capC = colorAt(capSq);
            remove(capC, PAWN, capSq);
            u.capturedPiece = PAWN;
            u.capturedColor = capC;

            remove(us,p,from);
            put(us,p,to);

            epSquare = -1; epMask = 0; epRemain = 0;

            advance_turn();
            return;
        }

        // --- Castling ---
        if (m.isCastle()){
            int side = (C[us][0].kTo == to) ? 0 : 1;
            const CastleGeom& G = C[us][side];

            remove(us, KING, G.kFrom);
            put(us, KING, G.kTo);

            remove(us, ROOK, G.rFrom);
            put(us, ROOK, G.rTo);

            clearK(us); clearQ(us);

            if (epSquare != -1 && (epMask & colorBit(us))) {
                if (--epRemain <= 0){ epSquare=-1; epMask=0; epRemain=0; }
            }

            advance_turn();
            return;
        }

        // --- Normal capture ---
        u.capturedPiece=-1; u.capturedColor=-1;
        if (m.capPiece()!=0 || m.capColor()<COLOR_N){
            Color capC=(Color)m.capColor();
            Piece capP=(Piece)m.capPiece();
            remove(capC,capP,to);
            revoke_rights_on_capture(capC, to);
            u.capturedPiece=capP;
            u.capturedColor=capC;
        }

        // Move piece
        remove(us,p,from);
        put(us,p,to);

        // New EP if double push
        if (p==PAWN && m.isDP()){
            int ep = MB_TO_SQ[SQ_TO_MB[to] - PAWN[us].push];
            epSquare = ep;
            epMask = 0;
            for (int c=0;c<COLOR_N;c++)
                if (teamOf((Color)c) != teamOf(us)) epMask |= colorBit((Color)c);
            epRemain = 2;
        } else {
            // decay EP if current mover is an opponent of the pawn that double-moved
            if (epSquare != -1 && (epMask & colorBit(us))){
                if (--epRemain <= 0){ epSquare = -1; epMask = 0; epRemain = 0; }
            }
        }

        advance_turn();
    }

    void undoMove(const Undo& u){
        const Move& m=u.move;
        sideToMove = (Color)u.stmPrev;

        // Restore EP state & rights
        epSquare = u.epSquarePrev;
        epMask   = u.epMaskPrev;
        epRemain = u.epRemainPrev;
        castleRights = u.castleRightsPrev;

        const Color us=sideToMove;
        const Piece p=(Piece)m.piece();

        if (m.isEP()){
            remove(us,p,m.to());
            put(us,p,m.from());
            int capSq = MB_TO_SQ[SQ_TO_MB[m.to()] - PAWN[us].push];
            put((Color)u.capturedColor, PAWN, capSq);
            std::memcpy(kingSq, u.kingSqPrev, sizeof(kingSq));
            return;
        }

        if (m.isCastle()){
            int side = (C[us][0].kTo == m.to()) ? 0 : 1;
            const CastleGeom& G = C[us][side];

            remove(us, ROOK, G.rTo);
            put(us, ROOK, G.rFrom);

            remove(us, KING, G.kTo);
            put(us, KING, G.kFrom);

            std::memcpy(kingSq, u.kingSqPrev, sizeof(kingSq));
            return;
        }

        // normal
        remove(us,p,m.to());
        put(us,p,m.from());
        if (u.capturedPiece!=-1)
            put((Color)u.capturedColor,(Piece)u.capturedPiece,m.to());

        std::memcpy(kingSq, u.kingSqPrev, sizeof(kingSq));
    }
};

// ---------------- Start position (Chess.com Teams) ----------------

void set_startpos(Board& b){
    b = Board(); // clear

    // Red back rank r=0, files 3..10: R N B Q K B N R
    const Piece order[8]={ROOK,KNIGHT,BISHOP,QUEEN,KING,BISHOP,KNIGHT,ROOK};
    for (int i=0;i<8;i++) b.put(RED, order[i], sqRF(0,3+i));
    for (int f=3; f<=10; ++f) b.put(RED, PAWN, sqRF(1,f));

    // Yellow r=13
    for (int i=0;i<8;i++) b.put(YELLOW, order[i], sqRF(13,3+i));
    for (int f=3; f<=10; ++f) b.put(YELLOW, PAWN, sqRF(12,f));

    // Blue file f=0, ranks 3..10 (top to bottom visually is 3..10)
    for (int i=0;i<8;i++) b.put(BLUE, order[i], sqRF(3+i,0));
    for (int r=3; r<=10; ++r) b.put(BLUE, PAWN, sqRF(r,1));

    // Green file f=13
    for (int i=0;i<8;i++) b.put(GREEN, order[i], sqRF(10-i,13)); // order along the file toward center
    for (int r=3; r<=10; ++r) b.put(GREEN, PAWN, sqRF(r,12));

    b.sideToMove = RED;
    b.castleRights = CR_K(RED)|CR_Q(RED)|CR_K(BLUE)|CR_Q(BLUE)|
                     CR_K(YELLOW)|CR_Q(YELLOW)|CR_K(GREEN)|CR_Q(GREEN);
    b.rebuild();
}

// ---------------- Perft ----------------

uint64_t perft(Board& b, int depth){
    if (depth==0) return 1ULL;
    std::vector<Move> moves;
    b.generateLegal(moves);
    if (depth==1) return (uint64_t)moves.size();

    uint64_t nodes=0;
    Undo u;
    for (const auto& m : moves){
        b.makeMove(m,u);
        nodes += perft(b, depth-1);
        b.undoMove(u);
    }
    return nodes;
}

void divide(Board& b, int depth){
    std::vector<Move> moves;
    b.generateLegal(moves);
    uint64_t total=0;
    Undo u;
    for (const auto& m : moves){
        b.makeMove(m,u);
        uint64_t n = perft(b, depth-1);
        b.undoMove(u);
        total += n;

        std::cout << std::setw(3) << m.from() << "->" << std::setw(3) << m.to()
                  << (m.isCastle()?" (O-O/O-O-O)":"")
                  << (m.isEP()?" (ep)":"")
                  << (m.isDP()?" (dp)":"")
                  << "  nodes=" << n << "\n";
    }
    std::cout << "Total: " << total << "\n";
}

// ---------------- Init once ----------------
struct InitOnce {
    InitOnce(){ init_geometry(); init_castling(); }
} _initOnce;

} // namespace FPC

// ---------------- Demo main ----------------
#ifdef FPC_PERFT_MAIN
int main(){
    using namespace FPC;
    Board b;
    set_startpos(b);

    for (int d=1; d<=3; ++d){
        uint64_t n = perft(b,d);
        std::cout << "perft("<<d<<") = " << n << "\n";
    }
    std::cout << "\nDivide depth 2:\n";
    divide(b,2);
    return 0;
}
#endif