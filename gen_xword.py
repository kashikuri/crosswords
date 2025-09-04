#!/usr/bin/env python3
"""
MxN crossword generator (entries shorter than or equal to max_len)

Usage:
  python gen_xword.py --dict short_answers.json --rows 5 --cols 5 --min-len 2 --max-len 4 --seed 0 \
    --symmetry rot180 --allow-repeats
"""
from __future__ import annotations
import argparse, random, sys, json
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Set

# ---------- Block-pattern generation ----------


def _runs_ok_row(bitmask: int, width: int, min_len: int, max_len: int) -> bool:
    run = 0
    for i in range(width + 1):
        bit = 1 if i == width else ((bitmask >> i) & 1)  # 1 = black, 0 = white
        if bit == 0:
            run += 1
        else:
            if run and not (min_len <= run <= max_len):
                return False
            run = 0
    return True


def _precompute_row_masks(width: int, min_len: int, max_len: int) -> List[int]:
    return [m for m in range(1 << width) if _runs_ok_row(m, width, min_len, max_len)]


def _col_partial_ok(rows: List[int], height: int, width: int, min_len: int, max_len: int) -> bool:
    h = len(rows)
    for c in range(width):
        run = 0
        for r in range(h):
            bit = (rows[r] >> c) & 1
            if bit == 0:
                run += 1
                if run > max_len:
                    return False
            else:
                if run and h < height and run < min_len:
                    return False
                run = 0
    return True


def _cols_final_ok(rows: List[int], height: int, width: int, min_len: int, max_len: int) -> bool:
    for c in range(width):
        run = 0
        for r in range(height + 1):
            bit = 1 if r == height else ((rows[r] >> c) & 1)
            if bit == 0:
                run += 1
            else:
                if run and not (min_len <= run <= max_len):
                    return False
                run = 0
    return True


def _bit_reverse(mask: int, width: int) -> int:
    out = 0
    for i in range(width):
        if (mask >> i) & 1:
            out |= 1 << (width - 1 - i)
    return out


def _ok_connectivity(cells: List[List[int]]) -> bool:
    R, C = len(cells), len(cells[0])
    start = None
    whites = 0
    for r in range(R):
        for c in range(C):
            if cells[r][c] == 0:
                whites += 1
                if start is None:
                    start = (r, c)
    if whites == 0:
        return False
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and cells[rr][cc] == 0 and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == whites


def gen_pattern(
    rows: int, cols: int, min_len: int, max_len: int, symmetry: str = "rot180", seed: Optional[int] = None
) -> List[List[int]]:
    rng = random.Random(seed)
    allowed = _precompute_row_masks(cols, min_len, max_len)
    row_masks: List[Optional[int]] = [None] * rows  # type: ignore

    def mirror_row(idx: int):
        if symmetry == "rot180":
            opp = rows - 1 - idx
            if row_masks[idx] is None:
                row_masks[opp] = None
            else:
                row_masks[opp] = _bit_reverse(row_masks[idx], cols)

    def rec(i: int) -> bool:
        if symmetry == "rot180" and i >= (rows + 1) // 2:
            if not _cols_final_ok(row_masks, rows, cols, min_len, max_len):
                return False
            grid = [[1 if (row_masks[r] >> c) & 1 else 0 for c in range(cols)] for r in range(rows)]
            return _ok_connectivity(grid)

        if symmetry == "none" and i == rows:
            if not _cols_final_ok(row_masks, rows, cols, min_len, max_len):
                return False
            grid = [[1 if (row_masks[r] >> c) & 1 else 0 for c in range(cols)] for r in range(rows)]
            return _ok_connectivity(grid)

        choices = allowed[:]
        rng.shuffle(choices)
        for mask in choices:
            row_masks[i] = mask
            if symmetry == "rot180":
                mirror_row(i)
            upto = i + 1 if symmetry == "none" else min(i + 1, rows)
            if _col_partial_ok(row_masks[:upto], rows, cols, min_len, max_len):
                if rec(i + 1):
                    return True
        row_masks[i] = None
        if symmetry == "rot180":
            mirror_row(i)
        return False

    if not rec(0):
        raise RuntimeError("Failed to build a valid pattern. Try different seed / lengths.")
    return [[1 if (row_masks[r] >> c) & 1 else 0 for c in range(cols)] for r in range(rows)]


# ---------- Slot detection & fill ----------


def _compute_slots(blocks: List[List[int]], min_len: int, max_len: int):
    R, C = len(blocks), len(blocks[0])
    slots = []
    # Across
    for r in range(R):
        c = 0
        while c < C:
            if blocks[r][c] == 0 and (c == 0 or blocks[r][c - 1] == 1):
                start = c
                while c < C and blocks[r][c] == 0:
                    c += 1
                L = c - start
                if min_len <= L <= max_len:
                    slots.append(("across", len(slots), [(r, cc) for cc in range(start, c)]))
            else:
                c += 1
    # Down
    for c in range(C):
        r = 0
        while r < R:
            if blocks[r][c] == 0 and (r == 0 or blocks[r - 1][c] == 1):
                start = r
                while r < R and blocks[r][c] == 0:
                    r += 1
                L = r - start
                if min_len <= L <= max_len:
                    slots.append(("down", len(slots), [(rr, c) for rr in range(start, r)]))
            else:
                r += 1
    return slots


def _filter_dict(words: List[str], L: int) -> List[str]:
    out = []
    for w in words:
        if len(w) == L and w.isalpha():
            out.append(w.upper())
    return out


def _pattern_for_slot(
    grid: List[List[Optional[str]]], coords: List[Tuple[int, int]]
) -> List[Optional[str]]:
    return [grid[r][c] for (r, c) in coords]


def _candidates_for_slot(slot, grid, words_by_len, used, allow_repeats: bool):
    _, _, coords = slot
    pats = _pattern_for_slot(grid, coords)
    L = len(coords)
    cands = []
    for w in words_by_len.get(L, []):
        if (not allow_repeats) and w in used:
            continue
        ok = True
        for i, ch in enumerate(w):
            if pats[i] is not None and pats[i] != ch:
                ok = False
                break
        if ok:
            cands.append(w)
    return cands


def fill_grid(
    blocks: List[List[int]],
    words: List[str],
    min_len: int,
    max_len: int,
    seed: Optional[int],
    allow_repeats: bool,
):
    rng = random.Random(seed)
    R, C = len(blocks), len(blocks[0])
    slots = _compute_slots(blocks, min_len, max_len)
    if not slots:
        raise RuntimeError("No slots found in pattern.")

    words_by_len = defaultdict(list)
    for L in range(min_len, max_len + 1):
        words_by_len[L] = _filter_dict(words, L)

    need_lens = set(len(s[2]) for s in slots)
    for L in need_lens:
        if not words_by_len[L]:
            raise RuntimeError(f"No dictionary words of length {L} after filtering.")

    grid: List[List[Optional[str]]] = [
        [None if blocks[r][c] == 0 else "#" for c in range(C)] for r in range(R)
    ]
    used: Set[str] = set()
    assign: Dict[int, str] = {}

    def select_unassigned():
        best_idx, best_cands, best_n = None, None, 10**9
        for sidx, slot in enumerate(slots):
            if sidx in assign:
                continue
            cands = _candidates_for_slot(slot, grid, words_by_len, used, allow_repeats)
            n = len(cands)
            if n == 0:
                return sidx, []
            if n < best_n:
                best_idx, best_cands, best_n = sidx, cands, n
        return best_idx, best_cands

    def put_word(sidx: int, word: str):
        _, _, coords = slots[sidx]
        for i, (r, c) in enumerate(coords):
            grid[r][c] = word[i]

    def remove_word(sidx: int):
        _, _, coords = slots[sidx]
        for r, c in coords:
            grid[r][c] = None
        for idx, word in assign.items():
            if idx == sidx:
                continue
            _, _, coords2 = slots[idx]
            for i, (r, c) in enumerate(coords2):
                grid[r][c] = word[i]

    def forward_ok() -> bool:
        for t, slot in enumerate(slots):
            if t in assign:
                continue
            if not _candidates_for_slot(slot, grid, words_by_len, used, allow_repeats):
                return False
        return True

    def backtrack() -> bool:
        if len(assign) == len(slots):
            return True
        sidx, cands = select_unassigned()
        if cands is None:
            return False
        rng.shuffle(cands)
        for w in cands:
            assign[sidx] = w
            put_word(sidx, w)
            if not allow_repeats:
                used.add(w)
            if forward_ok() and backtrack():
                return True
            if not allow_repeats and w in used:
                used.remove(w)
            del assign[sidx]
            remove_word(sidx)
        return False

    if not backtrack():
        raise RuntimeError("Backtracking failed. Try a different seed or a larger/cleaner dictionary.")
    return grid, slots, assign


# ---------- Output formatting ----------


def _pretty_blocks(blocks: List[List[int]]) -> str:
    return "\n".join("".join("#" if x else "." for x in row) for row in blocks)


def _pretty_grid(grid: List[List[Optional[str]]]) -> str:
    return "\n".join(" ".join(ch if ch else "." for ch in row) for row in grid)


def number_entries(blocks: List[List[int]], grid: List[List[Optional[str]]], word_clues: Dict[str, str]):
    R, C = len(blocks), len(blocks[0])
    num = 1
    across = []
    down = []
    # Across
    for r in range(R):
        c = 0
        while c < C:
            if blocks[r][c] == 0 and (c == 0 or blocks[r][c - 1] == 1):
                start = c
                word = []
                while c < C and blocks[r][c] == 0:
                    word.append(grid[r][c])
                    c += 1
                if len(word) >= 2:
                    sol = "".join(word)
                    across.append((num, r, start, sol, word_clues.get(sol, "")))
                    num += 1
            else:
                c += 1
    # Down
    for c in range(C):
        r = 0
        while r < R:
            if blocks[r][c] == 0 and (r == 0 or blocks[r - 1][c] == 1):
                start = r
                word = []
                while r < R and blocks[r][c] == 0:
                    word.append(grid[r][c])
                    r += 1
                if len(word) >= 2:
                    sol = "".join(word)
                    down.append((num, start, c, sol, word_clues.get(sol, "")))
                    num += 1
            else:
                r += 1
    return across, down


# ---------- Main ----------


def main(args=None):
    # If args is None, provide defaults for direct execution in an editor
    if args is None:

        class Args:
            dict = "short_answers.json"
            rows = 5
            cols = 5
            min_len = 3
            max_len = 5
            seed = 0
            symmetry = "rot180"
            allow_repeats = False

        args = Args()
    else:
        ap = argparse.ArgumentParser(
            description="Generate an MxN crossword with all entries within a length range."
        )
        ap.add_argument("--dict", required=True, help="Path to JSON word+clue list.")
        ap.add_argument("--rows", type=int, default=5, help="Grid rows (default 5).")
        ap.add_argument("--cols", type=int, default=5, help="Grid cols (default 5).")
        ap.add_argument("--min-len", type=int, default=2, help="Minimum entry length (default 2).")
        ap.add_argument("--max-len", type=int, default=4, help="Maximum entry length (default 4).")
        ap.add_argument("--seed", type=int, default=None, help="Seed for RNG.")
        ap.add_argument("--symmetry", choices=["none", "rot180"], default="rot180", help="Block symmetry.")
        ap.add_argument(
            "--allow-repeats", action="store_true", help="Allow reusing the same word in multiple slots."
        )
        args = ap.parse_args(args)

    # Validation checks
    if args.rows < 2 or args.cols < 2:
        print("rows and cols must be >= 2.", file=sys.stderr)
        sys.exit(2)
    if not (1 <= args.min_len <= args.max_len):
        print("Require 1 <= min_len <= max_len.", file=sys.stderr)
        sys.exit(2)
    if args.max_len > max(args.rows, args.cols):
        print("max_len cannot exceed the larger grid dimension.", file=sys.stderr)
        sys.exit(2)

    # Load JSON dictionary and remove duplicate answers
    with open(args.dict, "r", encoding="utf-8", errors="ignore") as f:
        raw_json = json.load(f)

    seen_answers = set()
    unique_entries = []
    for entry in raw_json:
        if entry and entry[0].isalpha():
            answer = entry[0].strip().upper()
            clue = entry[1].strip()
            if answer not in seen_answers:
                seen_answers.add(answer)
                unique_entries.append((answer, clue))

    words = [answer for answer, _ in unique_entries]
    word_clues = {answer: clue for answer, clue in unique_entries}

    # Generate pattern and fill grid
    blocks = gen_pattern(
        rows=args.rows,
        cols=args.cols,
        min_len=args.min_len,
        max_len=args.max_len,
        symmetry=args.symmetry,
        seed=args.seed,
    )

    grid, slots, assign = fill_grid(
        blocks, words, args.min_len, args.max_len, seed=args.seed, allow_repeats=args.allow_repeats
    )

    # Print output
    print("\n# Pattern (#=block, .=white)")
    print(_pretty_blocks(blocks))
    print("\n# Filled grid")
    print(_pretty_grid(grid))

    across, down = number_entries(blocks, grid, word_clues)

    print("\nACROSS")
    for num, r, c, sol, clue in across:
        print(f"{num}. ({r+1},{c+1})  {sol}  -- {clue}")
        # print(f"{num}. ({r+1},{c+1})  {clue}")

    print("\nDOWN")
    for num, r, c, sol, clue in down:
        print(f"{num}. ({r+1},{c+1})  {sol}  -- {clue}")
        # print(f"{num}. ({r+1},{c+1})  {clue}")


if __name__ == "__main__":
    main()
