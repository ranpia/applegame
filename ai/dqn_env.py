import numpy as np
from utils.group_generator import generate_all_valid_groups

class FruitBoxEnv:
    def __init__(self, board):
        self.original_board = board.copy()
        self.board = board.copy()
        self.done = False

    def reset(self, new_board=None):
        if new_board is not None:
            self.original_board = new_board.copy()
        self.board = self.original_board.copy()
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.board[np.newaxis, :, :]  # shape: (1, 10, 17)

    def get_action_list(self):
        return generate_all_valid_groups(self.board)

    def step(self, group):
        """
        group: List of (r, c) coordinates that sum to 10
        """
        if self.done:
            raise Exception("환경이 종료되었습니다. reset()을 호출하세요.")

        values = [self.board[r, c] for r, c in group if self.board[r, c] not in (0, -1)]
        if sum(values) != 10:
            reward = -1  # 잘못된 선택
            done = True
        else:
            reward = len(values)
            for r, c in group:
                self.board[r, c] = 0  # 사과 제거
            done = len(self.get_action_list()) == 0

        self.done = done
        return self._get_state(), reward, done

# 테스트용
if __name__ == '__main__':
    board = np.random.randint(1, 9, (10, 17))
    env = FruitBoxEnv(board)
    state = env.reset()
    actions = env.get_action_list()
    print(f"유효 그룹 수: {len(actions)}")
    if actions:
        next_state, reward, done = env.step(actions[0])
        print("보상:", reward, "종료?:", done)