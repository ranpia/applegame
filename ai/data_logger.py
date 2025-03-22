import csv
import os
import time

def save_training_data(board, mask, path="data/training_from_board.csv"):
    import os, csv
    flatten_board = board.flatten().tolist()
    flatten_mask = mask.flatten().tolist()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [f"x{i}" for i in range(170)] + [f"m{i}" for i in range(170)]
            writer.writerow(header)
        writer.writerow(flatten_board + flatten_mask)



def log_score(score, drag_count, path="data/score_log.txt"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(f"[{timestamp}] 점수: {score}점 | 드래그: {drag_count}회\n")