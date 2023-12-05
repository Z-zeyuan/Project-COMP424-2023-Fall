# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


def check_valid_step(self, start_pos, end_pos, barrier_dir):
    """
    Check if the step the agent takes is valid (reachable and within max steps).

    Parameters
    ----------
    start_pos : tuple
        The start position of the agent.
    end_pos : np.ndarray
        The end position of the agent.
    barrier_dir : int
        The direction of the barrier.
    """
    # Endpoint already has barrier or is border
    r, c = end_pos
    if self.chess_board[r, c, barrier_dir]:
        return False
    if np.array_equal(start_pos, end_pos):
        return True

    # Get position of the adversary
    adv_pos = self.p0_pos if self.turn else self.p1_pos

    # BFS
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == self.max_step:
            break
        for dir, move in enumerate(self.moves):
            if self.chess_board[r, c, dir]:
                continue

            next_pos = cur_pos + move
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached


def check_endgame(self):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    # Union-Find
    father = dict()
    for r in range(self.board_size):
        for c in range(self.board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(self.board_size):
        for c in range(self.board_size):
            for dir, move in enumerate(
                    self.moves[1:3]
            ):  # Only check down and right
                if self.chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(self.board_size):
        for c in range(self.board_size):
            find((r, c))
    p0_r = find(tuple(self.p0_pos))
    p1_r = find(tuple(self.p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    player_win = None
    win_blocks = -1
    if p0_score > p1_score:
        player_win = 0
        win_blocks = p0_score
    elif p0_score < p1_score:
        player_win = 1
        win_blocks = p1_score
    else:
        player_win = -1  # Tie
    if player_win >= 0:
        logging.info(
            f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
        )
    else:
        logging.info("Game ends! It is a Tie!")
    return True, p0_score, p1_score


def check_boundary(self, pos):
    r, c = pos
    return 0 <= r < self.board_size and 0 <= c < self.board_size


def set_barrier(self, r, c, dir):
    # Set the barrier to True
    self.chess_board[r, c, dir] = True
    # Set the opposite barrier to True
    move = self.moves[dir]
    self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

def remove_barrier(chess_board, x, y, direction):
    return

def random_walk(self, my_pos, adv_pos):
    """
    Randomly walk to the next position in the board.

    Parameters
    ----------
    my_pos : tuple
        The position of the agent.
    adv_pos : tuple
        The position of the adversary.
    """
    steps = np.random.randint(0, self.max_step + 1)

    # Pick steps random but allowable moves
    for _ in range(steps):
        r, c = my_pos

        # Build a list of the moves we can make
        allowed_dirs = [d
                        for d in range(0, 4)  # 4 moves possible
                        if not self.chess_board[r, c, d] and  # chess_board True means wall
                        not adv_pos == (r + self.moves[d][0], c + self.moves[d][1])]  # cannot move through Adversary

        if len(allowed_dirs) == 0:
            # If no possible move, we must be enclosed by our Adversary
            break

        random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

        # This is how to update a row,col by the entries in moves
        # to be consistent with game logic
        m_r, m_c = self.moves[random_dir]
        my_pos = (r + m_r, c + m_c)

    # Final portion, pick where to put our new barrier, at random
    r, c = my_pos
    # Possibilities, any direction such that chess_board is False
    allowed_barriers = [i for i in range(0, 4) if not self.chess_board[r, c, i]]
    # Sanity check, no way to be fully enclosed in a square, else game already ended
    assert len(allowed_barriers) >= 1
    dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

    return my_pos, dir


class MCTS():
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.chess_board = chess_board
        self.board_size = len(self.chess_board)
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

        self.children = []

    def get_move(self):

        moves = []
        (r, c), dir = random_walk(self.chess_board, self.my_pos, self.adv_pos)
        moves.append(((r, c), dir))
        for i in range(depth):#depth to be tuned
            (r, c), dir = random_walk(self.chess_board, self.my_pos, self.adv_pos, self.max_step)
            #set_barrier(self.chess_board, r, c, dir)
            #check small moves

            if self.chess_board[self.adv_pos[0], self.adv_pos[1]].sum() >= 2: return ((r, c), dir)
            moves.append(((r, c), dir))
            #remove_barrier(self.chess_board, r, c, dir)

        #find best move
        l = []
        for move in moves:
            (r, c), dir = move
            set_barrier(self.chess_board, r, c, dir)
            #find huristic
            remove_barrier(self.chess_board, r, c, dir)

        #sort l
        return l[0][0]

    def expand_children(self):
        moves = []
        i = 0
        #num_child = self.board_size
        #num_simul = num_child
        while i < num_child:
            move = self.get_move()
            (x, y), direction = move
            if move in moves: continue
            set_barrier(self.chess_board, x, y, direction)
            #get huristic of move
            remove_barrier(self.chess_board, x, y, direction)
            self.children.append((huristic, move))
            i += 1

    def find_move(self):
        # choose the move with the highest utility
        self.expand_children()
        #sort children
        return self.children[0][1]


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper 0functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
