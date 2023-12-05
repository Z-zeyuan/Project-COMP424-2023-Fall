from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math
import random

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Our agent which uses MCTS coupled with a simple heuristic function
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True

    def __str__(self) -> str:
        return self.name

    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        root_state = WorldState(chess_board, my_pos, adv_pos, max_step)
        mcts = MCTS(root_state)

        move = mcts.run(1000)

        return move


class WorldState:

    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

        self.main_agent_turn = True # Means my_pos corresponds to our agent not the adversary

        self.moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.board_size = self.chess_board.shape[0]

    def get_possible_moves(self):
        """Get a list of all possible moves in the given state"""
        possible_moves = []

        visited = set()
        queue = [(self.my_pos, 0)]

        while queue:
            current, steps = queue.pop(0)
            if current in visited or steps > self.max_step:
                continue

            r, c = current

            visited.add(current)
            possible_moves.extend([(current, d) for d in range(0,4) if not self.chess_board[r,c,d]])

            for d in range(0,4):
                new_position = (r + self.moves[d][0], c + self.moves[d][1])
                if not self.chess_board[r,c,d] and not self.adv_pos == new_position and new_position not in visited:
                    queue.append((new_position, steps + 1))

        return possible_moves
    

    def check_endgame(self):
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
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        final_score = p0_score - p1_score if self.main_agent_turn else p1_score - p0_score
        if p0_r == p1_r:
            return False, final_score
        return True, final_score
    

    def get_next_state(self, move):
        # Assuming the move is legal (should always be the case for our purposes)
        next_state = deepcopy(self)

        new_pos, new_wall = move

        # Move player to position and place wall
        next_state.my_pos = new_pos
        next_state.set_barrier(new_pos[0], new_pos[1], new_wall)

        # Switch whose turn to play it is
        next_state.my_pos, next_state.adv_pos = next_state.adv_pos, next_state.my_pos
        next_state.main_agent_turn = not next_state.main_agent_turn

        return next_state
    
    def set_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True


    def get_random_move(self):
        return random.choice(self.get_possible_moves())


class Node:

    def __init__(self, state, parent=None, played_move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.played_move = played_move

        self.untried_moves = self.state.get_possible_moves()

    def __str__(self):
        return f"Node: State: my_pos={self.state.my_pos}, adv_pos={self.state.adv_pos}"

    def is_leaf_node(self):
        return bool(self.children)
    
    def is_terminal_node(self):
        return self.state.check_endgame()[0]
    
    def select_child(self, c=1.3):
        children_ucts = [
            (child.wins / child.visits) + c * np.sqrt((2 * np.log(self.visits))/child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(children_ucts)]

    def fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def get_untried_move(self):
        return self.untried_moves.pop(random.randint(0,len(self.untried_moves)-1))




class MCTS:

    def __init__(self, root_state):
        self.root = Node(root_state)

        self.counter = 0

    def run(self, num_simulations):
        for _ in range(num_simulations):
            leaf = self.select(self.root)
            result = self.simulate(leaf)
            self.backpropagate(leaf, result)

        return self.root.select_child().played_move

    def select(self, node):
        curr_node = node
        while not curr_node.is_terminal_node():
            if not curr_node.fully_expanded():
                self.counter += 1
                return self.expand(curr_node)
            else:
                curr_node = curr_node.select_child()
        return curr_node

    def expand(self, node):
        new_move = node.get_untried_move()
        new_state = node.state.get_next_state(new_move)
        new_node = Node(new_state, parent=node, played_move=new_move)
        node.children.append(new_node)
        return new_node

    def simulate(self, node, rollout_depth=10):
        rollout_state = node.state
        ended, score = rollout_state.check_endgame()
        while not ended:
            move = rollout_state.get_random_move()
            rollout_state = rollout_state.get_next_state(move)
            ended, score = rollout_state.check_endgame()
        return score

    def backpropagate(self, node, result):
        curr_node = node
        while curr_node is not None:
            curr_node.visits += 1
            curr_node.wins += result
            curr_node = curr_node.parent
