import numpy as np
import itertools

def generate_all_valid_groups(board):
    """
    현재 보드에서 합이 10이 되는 유효한 그룹을 모두 반환.
    그룹은 [(r, c), (r, c), ...] 형식의 리스트.
    """
    valid_groups = []
    rows, cols = board.shape
    for r1 in range(rows):
        for c1 in range(cols):
            if board[r1, c1] in (0, -1):
                continue
            for r2 in range(rows):
                for c2 in range(cols):
                    if (r2 == r1 and c2 == c1) or board[r2, c2] in (0, -1):
                        continue
                    for r3 in range(rows):
                        for c3 in range(cols):
                            coords = [(r1, c1), (r2, c2), (r3, c3)]
                            try:
                                vals = [board[r, c] for r, c in coords]
                                if sum(vals) == 10:
                                    coords = sorted(set(coords))
                                    if coords not in valid_groups:
                                        valid_groups.append(coords)
                            except IndexError:
                                continue
    return valid_groups