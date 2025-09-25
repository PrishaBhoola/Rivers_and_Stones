
import random
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import copy

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows


def score_cols_for(cols: int) -> List[int]:
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))


def top_score_row() -> int:
    return 2


def bottom_score_row(rows: int) -> int:
    return rows - 3


def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)


def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)


def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"


def generate_all_moves_basic(board, player, rows, cols, score_cols):
    moves = []

    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p or p.owner != player:
                continue

            if is_own_score_cell(x, y, player, rows, cols, score_cols):
                if getattr(p, "side", "stone") == "river":
                    moves.append({"action": "flip", "from": [x, y], "orientation": None})
                continue

            valid_moves = compute_valid_moves(board, x, y, player, rows, cols, score_cols)
            for move_coord in valid_moves['moves']:
                moves.append({"action": "move", "from": [x, y], "to": list(move_coord)})

            for push_data in valid_moves['pushes']:
                target_coord, dest_coord = push_data
                moves.append({
                    "action": "push",
                    "from": [x, y],
                    "to": list(target_coord),
                    "pushed_to": list(dest_coord)
                })

            if getattr(p, "side", "stone") == "stone":
                moves.append({"action": "flip", "from": [x, y], "orientation": "horizontal"})
                moves.append({"action": "flip", "from": [x, y], "orientation": "vertical"})
            else:
                moves.append({"action": "flip", "from": [x, y], "orientation": None})
                new_orient = "vertical" if p.orientation == "horizontal" else "horizontal"
                moves.append({"action": "rotate", "from": [x, y], "orientation": new_orient})

    return moves


def compute_valid_moves(board, sx, sy, player, rows, cols, score_cols):
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    moves = set()
    pushes = []

    if not in_bounds(sx, sy, rows, cols):
        return {'moves': moves, 'pushes': pushes}

    p = board[sy][sx]
    if not p or p.owner != player:
        return {'moves': moves, 'pushes': pushes}

    for dx, dy in dirs:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue

        target = board[ty][tx]

        if not target:
            moves.add((tx, ty))
        elif getattr(target, "side", "stone") == "river":
            flow = get_river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for dest in flow:
                moves.add(dest)
        elif getattr(target, "side", "stone") == "stone":
            if getattr(p, "side", "stone") == "stone":
                px, py = tx + dx, ty + dy
                if (in_bounds(px, py, rows, cols) and
                        not board[py][px] and
                        not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    pushes.append(((tx, ty), (px, py)))
            else:
                pushed_player = target.owner
                flow = get_river_flow_destinations(board, tx, ty, sx, sy, pushed_player, rows, cols, score_cols, True)
                for dest in flow:
                    if not is_opponent_score_cell(dest[0], dest[1], player, rows, cols, score_cols):
                        pushes.append(((tx, ty), dest))

    return {'moves': moves, 'pushes': pushes}


def get_river_flow_destinations(board, rx, ry, sx, sy, player, rows, cols, score_cols, river_push=False):
    destinations = []
    visited = set()
    q = [(rx, ry)]

    while q:
        x, y = q.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[sy][sx] if river_push and x == rx and y == ry else board[y][x]

        if not cell:
            if not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                destinations.append((x, y))
            continue

        if getattr(cell, "side", "stone") != "river":
            continue

        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                next_cell = board[ny][nx]
                if not next_cell:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue

                if getattr(next_cell, "side", "stone") == "river":
                    q.append((nx, ny))
                break

    out = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            out.append(d)

    return out

def agent_apply_move(board, move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[
    bool, str]:
    action = move.get("action")

    if action == "move":
        return _apply_move_action(board, move, player, rows, cols, score_cols)
    elif action == "push":
        return _apply_push_action(board, move, player, rows, cols, score_cols)
    elif action == "flip":
        return _apply_flip_action(board, move, player, rows, cols, score_cols)
    elif action == "rotate":
        return _apply_rotate_action(board, move, player, rows, cols, score_cols)

    return False, "unknown action"
def _apply_move_action(board, move, player, rows, cols, score_cols):
    fr = move.get("from")
    to = move.get("to")
    if not fr or not to:
        return False, "bad move format"

    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])

    if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
        return False, "out of bounds"

    if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
        return False, "cannot move into opponent score cell"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if board[ty][tx] is None:
        board[ty][tx] = piece
        board[fy][fx] = None
        return True, "moved"

    pushed_to = move.get("pushed_to")
    if not pushed_to:
        return False, "destination occupied; pushed_to required"

    ptx, pty = int(pushed_to[0]), int(pushed_to[1])
    dx, dy = tx - fx, ty - fy

    if (ptx, pty) != (tx + dx, ty + dy):
        return False, "invalid pushed_to"

    if not in_bounds(ptx, pty, rows, cols):
        return False, "pushed_to out of bounds"

    if is_opponent_score_cell(ptx, pty, player, rows, cols, score_cols):
        return False, "cannot push into opponent score"

    if board[pty][ptx] is not None:
        return False, "pushed_to not empty"

    board[pty][ptx] = board[ty][tx]
    board[ty][tx] = piece
    board[fy][fx] = None
    return True, "moved with push"
def _apply_push_action(board, move, player, rows, cols, score_cols):
    fr = move.get("from")
    to = move.get("to")
    pushed_to = move.get("pushed_to")

    if not fr or not to or not pushed_to:
        return False, "bad push format"

    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])
    px, py = int(pushed_to[0]), int(pushed_to[1])

    if (is_opponent_score_cell(tx, ty, player, rows, cols, score_cols) or
            is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
        return False, "push would move into opponent score cell"

    if not (in_bounds(fx, fy, rows, cols) and
            in_bounds(tx, ty, rows, cols) and
            in_bounds(px, py, rows, cols)):
        return False, "out of bounds"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if board[ty][tx] is None:
        return False, "'to' must be occupied"

    if board[py][px] is not None:
        return False, "pushed_to not empty"

    board[py][px] = board[ty][tx]
    board[ty][tx] = board[fy][fx]
    board[fy][fx] = None
    return True, "pushed"
def _apply_flip_action(board, move, player, rows, cols, score_cols):
    fr = move.get("from")
    if not fr:
        return False, "bad flip format"

    fx, fy = int(fr[0]), int(fr[1])
    piece = board[fy][fx]

    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if piece.side == "stone":
        orientation = move.get("orientation")
        if orientation not in ("horizontal", "vertical"):
            return False, "stone->river needs orientation"

        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)

        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None

        for dx, dy in flow:
            if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
                return False, "flip would allow flow into opponent score cell"

        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        return True, "flipped to river"
    else:
        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None
        return True, "flipped to stone"
def _apply_rotate_action(board, move, player, rows, cols, score_cols):
    fr = move.get("from")
    if not fr:
        return False, "bad rotate format"

    fx, fy = int(fr[0]), int(fr[1])
    piece = board[fy][fx]

    if piece is None or piece.owner != player or piece.side != "river":
        return False, "invalid rotate"

    old_orientation = piece.orientation
    piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"

    flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)

    for dx, dy in flow:
        if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
            piece.orientation = old_orientation
            return False, "rotate would allow flow into opponent score cell"

    return True, "rotated"


class BaseAgent(ABC):
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        # to override
        return None
    def generate_all_moves(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        return generate_all_moves_basic(board, self.player, rows, cols, score_cols)
    def evaluate_board(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> float:
        # to override
        return 0.0
    def simulate_move(self, board: List[List[Any]], move: Dict[str, Any], rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
        board_copy = copy.deepcopy(board)
        success, message = agent_apply_move(board_copy, move, self.player, rows, cols, score_cols)
        if success:
            return True, board_copy
        else:
            return False, message

def manhattan_distance_to_score(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    target_row = bottom_score_row(rows) if player == "square" else top_score_row()
    min_dist = float('inf')
    for sc in score_cols:
        dist = abs(x - sc) + abs(y - target_row)
        min_dist = min(min_dist, dist)
    return min_dist


def generate_all_moves(board, player, rows, cols, score_cols):
    moves = []

    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p or p.owner != player:
                continue

            if is_own_score_cell(x, y, player, rows, cols, score_cols):
                if getattr(p, "side", "stone") == "river":
                    moves.append({"action": "flip", "from": [x, y], "orientation": None})
                continue

            valid_moves = compute_valid_moves(board, x, y, player, rows, cols, score_cols)
            for move_coord in valid_moves['moves']:
                moves.append({"action": "move", "from": [x, y], "to": list(move_coord)})

            for push_data in valid_moves['pushes']:
                target_coord, dest_coord = push_data
                moves.append({
                    "action": "push",
                    "from": [x, y],
                    "to": list(target_coord),
                    "pushed_to": list(dest_coord)
                })

            if getattr(p, "side", "stone") == "stone":
                moves.append({"action": "flip", "from": [x, y], "orientation": "horizontal"})
                moves.append({"action": "flip", "from": [x, y], "orientation": "vertical"})
            else:
                moves.append({"action": "flip", "from": [x, y], "orientation": None})
                new_orient = "vertical" if p.orientation == "horizontal" else "horizontal"
                moves.append({"action": "rotate", "from": [x, y], "orientation": new_orient})

    return moves


def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    count = 0
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    return count


def _get_shortest_path_to_score(start_pos: Tuple[int, int], player: str, board: List[List[Any]], rows: int, cols: int,
                                score_cols: List[int]) -> int:
    from collections import deque
    q = deque([(start_pos, 0)])
    visited = {start_pos}
    score_row = bottom_score_row(rows) if player == "square" else top_score_row()
    target_cells = set((sc, score_row) for sc in score_cols)
    while q:
        (x, y), dist = q.popleft()
        if (x, y) in target_cells:
            return dist
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols) or (nx, ny) in visited:
                continue
            target_cell = board[ny][nx]
            if target_cell is None or (target_cell.side == "river" and target_cell.owner == player):
                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))
            if target_cell and target_cell.side == "river":
                flow_dests = get_river_flow_destinations(board, nx, ny, x, y, player, rows, cols, score_cols)
                for dest_x, dest_y in flow_dests:
                    if (dest_x, dest_y) not in visited:
                        visited.add((dest_x, dest_y))
                        q.append(((dest_x, dest_y), dist + 1))
    return float('inf')


def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    opponent = get_opponent(player)
    score = 0.0
    weights = {
        'stone_in_score': 100000000000,
        'proximity': 500000000,
        'very_close_bonus': 200000,
        'close_bonus': 80000,
        'blocking_piece': 15,
        'river_count': -0.2,
        'river_reach': 0.15,
        'stone_river_access': 3,
        'ortho_link': 2.0,
        'para_link': 0,
        'centrality': 1.5,
        'too_few_stones_penalty': -100000
    }
    player_stones = []
    opponent_stones = []
    player_rivers = []
    opponent_rivers = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece:
                if piece.owner == player:
                    if piece.side == 'stone':
                        player_stones.append((x, y))
                    else:
                        player_rivers.append((x, y))
                else:
                    if piece.side == 'stone':
                        opponent_stones.append((x, y))
                    else:
                        opponent_rivers.append((x, y))

    player_score_count = sum(
        1 for x, y in player_stones if is_opponent_score_cell(x, y, opponent, rows, cols, score_cols))
    opponent_score_count = sum(
        1 for x, y in opponent_stones if is_opponent_score_cell(x, y, player, rows, cols, score_cols))
    score += player_score_count * weights['stone_in_score']
    score -= opponent_score_count * weights['stone_in_score']

    very_close_player = 0
    close_player = 0
    very_close_opponent = 0
    close_opponent = 0

    for x, y in player_stones:
        if not is_opponent_score_cell(x, y, opponent, rows, cols, score_cols):
            dist = manhattan_distance_to_score(x, y, player, rows, cols, score_cols)
            if dist == 1:
                very_close_player += 1
            elif dist == 2:
                close_player += 1

    for x, y in opponent_stones:
        if not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
            dist = manhattan_distance_to_score(x, y, opponent, rows, cols, score_cols)
            if dist == 1:
                very_close_opponent += 1
            elif dist == 2:
                close_opponent += 1

    score += (very_close_player - very_close_opponent) * weights['very_close_bonus']
    score += (close_player - close_opponent) * weights['close_bonus']

    player_distances = sorted(
        [_get_shortest_path_to_score(pos, player, board, rows, cols, score_cols) for pos in player_stones])[:4]
    opponent_distances = sorted(
        [_get_shortest_path_to_score(pos, opponent, board, rows, cols, score_cols) for pos in opponent_stones])[:4]
    for d in player_distances:
        if d != float('inf'):
            score -= d * weights['proximity']
    for d in opponent_distances:
        if d != float('inf'):
            score += d * weights['proximity']

    player_score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
    for sc in score_cols:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = sc + dx, player_score_row + dy
            if in_bounds(nx, ny, rows, cols) and board[ny][nx] and board[ny][nx].owner == player:
                score += weights['blocking_piece']

    score += (len(player_rivers) - len(opponent_rivers)) * weights['river_count']
    river_reach_player = 0
    river_reach_opponent = 0
    stone_access_player = 0
    stone_access_opponent = 0
    player_active_stones = [(x, y) for x, y in player_stones if
                            not is_opponent_score_cell(x, y, opponent, rows, cols, score_cols)]
    opponent_active_stones = [(x, y) for x, y in opponent_stones if
                              not is_opponent_score_cell(x, y, player, rows, cols, score_cols)]
    for x, y in player_rivers:
        reach = len(get_river_flow_destinations(board, x, y, x, y, player, rows, cols, score_cols))
        river_reach_player += reach
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny, rows, cols) and (nx, ny) in player_active_stones:
                stone_access_player += reach
    for x, y in opponent_rivers:
        reach = len(get_river_flow_destinations(board, x, y, x, y, opponent, rows, cols, score_cols))
        river_reach_opponent += reach
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny, rows, cols) and (nx, ny) in opponent_active_stones:
                stone_access_opponent += reach
    score += (river_reach_player - river_reach_opponent) * weights['river_reach']
    score += (stone_access_player - stone_access_opponent) * weights['stone_river_access']
    ortho_links_player, para_links_player = 0, 0
    for x, y in player_rivers:
        adjacent_to_scored_stone = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (in_bounds(nx, ny, rows, cols) and
                    (nx, ny) in player_stones and
                    is_opponent_score_cell(nx, ny, opponent, rows, cols, score_cols)):
                adjacent_to_scored_stone = True
                break
        if adjacent_to_scored_stone:
            continue
        p = board[y][x]
        flow_dx, flow_dy = (1, 0) if p.orientation == 'horizontal' else (0, 1)
        for sign in [-1, 1]:
            nx, ny = x + (flow_dx * sign), y + (flow_dy * sign)
            if in_bounds(nx, ny, rows, cols) and (nx, ny) in player_rivers:
                neighbor = board[ny][nx]
                if neighbor.orientation == p.orientation:
                    para_links_player += 1
                else:
                    ortho_links_player += 1
    ortho_links_opp, para_links_opp = 0, 0
    for x, y in opponent_rivers:
        adjacent_to_scored_stone = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (in_bounds(nx, ny, rows, cols) and
                    (nx, ny) in opponent_stones and
                    is_opponent_score_cell(nx, ny, player, rows, cols, score_cols)):
                adjacent_to_scored_stone = True
                break
        if adjacent_to_scored_stone:
            continue
        p = board[y][x]
        flow_dx, flow_dy = (1, 0) if p.orientation == 'horizontal' else (0, 1)
        for sign in [-1, 1]:
            nx, ny = x + (flow_dx * sign), y + (flow_dy * sign)
            if in_bounds(nx, ny, rows, cols) and (nx, ny) in opponent_rivers:
                neighbor = board[ny][nx]
                if neighbor.orientation == p.orientation:
                    para_links_opp += 1
                else:
                    ortho_links_opp += 1
    score += (ortho_links_player - ortho_links_opp) * weights['ortho_link']
    score += (para_links_player - para_links_opp) * weights['para_link']
    center_x, center_y = cols / 2, rows / 2
    player_centrality = sum(1 / (1 + abs(x - center_x) + abs(y - center_y)) for x, y in player_stones)
    opponent_centrality = sum(1 / (1 + abs(x - center_x) + abs(y - center_y)) for x, y in opponent_stones)
    score += (player_centrality - opponent_centrality) * weights['centrality']
    min_stones = 6
    if len(player_stones) < min_stones:
        score += (len(player_stones) - min_stones) * weights['too_few_stones_penalty']
    return score


def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int,
                  score_cols: List[int]) -> Tuple[bool, Any]:
    try:
        from gameEngine import validate_and_apply_move
        board_copy = copy.deepcopy(board)
        success, message = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols)
        return success, board_copy if success else message
    except ImportError:
        return True, copy.deepcopy(board)


class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.move_history = []
        self.history_size = 5

    def _update_move_history(self, move):
        self.move_history.append(move)
        if len(self.move_history) > self.history_size:
            self.move_history.pop(0)

    def _is_repetitive_move(self, move):
        if not self.move_history:
            return False
        action = move.get('action')
        from_pos = move.get('from')
        if not from_pos:
            return False
        if action in ['flip', 'rotate']:
            recent_count = 0
            for hist_move in reversed(self.move_history):
                hist_action = hist_move.get('action')
                hist_from = hist_move.get('from')
                if hist_from == from_pos and (
                        (action == 'flip' and hist_action == 'flip') or
                        (action == 'rotate' and hist_action == 'rotate')
                ):
                    recent_count += 1
                    if recent_count >= 2:
                        return True
        if action == 'move':
            to_pos = move.get('to')
            if len(self.move_history) >= 2:
                prev_move = self.move_history[-1]
                if prev_move.get('action') == 'move':
                    prev_from = prev_move.get('from')
                    prev_to = prev_move.get('to')
                    if from_pos == prev_to and to_pos == prev_from:
                        return True
                    if len(self.move_history) >= 4:
                        for i in range(len(self.move_history) - 2):
                            old_move = self.move_history[i]
                            if (old_move.get('action') == 'move' and
                                    old_move.get('from') == from_pos and
                                    old_move.get('to') == to_pos):
                                return True
        if action == 'push':
            to_pos = move.get('to')
            pushed_to = move.get('pushed_to')
            if to_pos and pushed_to and len(self.move_history) >= 2:
                for hist_move in self.move_history[-2:]:
                    if hist_move.get('action') == 'push':
                        hist_from = hist_move.get('from')
                        hist_to = hist_move.get('to')
                        hist_pushed_to = hist_move.get('pushed_to')
                        if (hist_pushed_to == from_pos and hist_to == to_pos) or \
                                (hist_from == pushed_to and hist_to == from_pos):
                            return True
                if len(self.move_history) >= 4:
                    for i in range(len(self.move_history) - 3):
                        old_move = self.move_history[i]
                        if (old_move.get('action') == 'push' and
                                old_move.get('from') == from_pos and
                                old_move.get('to') == to_pos):
                            return True
        return False

    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float,
               opponent_time: float) -> Optional[Dict[str, Any]]:
        player_stones = []
        player_rivers = []
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if piece and piece.owner == self.player:
                    if piece.side == 'stone':
                        player_stones.append((x, y))
                    else:
                        player_rivers.append((x, y))
        for river in player_rivers:
            # if river in scoring area, turn it to stone
            if is_own_score_cell(river[0], river[1], self.player, rows, cols, score_cols):
                move = {"action": "flip", "from": [river[0], river[1]]}
                success, new_board = simulate_move(board, move, self.player, rows, cols, score_cols)
                if success:
                    self._update_move_history(move)
                    return move
        moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        if not moves:
            return None
        current_stones_in_score = count_stones_in_scoring_area(board, self.player, rows, cols, score_cols)
        for move in moves:
            success, new_board = simulate_move(board, move, self.player, rows, cols, score_cols)
            if not success:
                continue
            stones_in_score = count_stones_in_scoring_area(new_board, self.player, rows, cols, score_cols)
            if stones_in_score > current_stones_in_score:
                return move
        stone_distances = []
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if piece and piece.owner == self.player and piece.side == 'stone':
                    dist = _get_shortest_path_to_score((x, y), self.player, board, rows, cols, score_cols)
                    if dist != float('inf'):
                        stone_distances.append(((x, y), dist))
        stone_distances.sort(key=lambda x: x[1])
        if stone_distances and stone_distances[0][1] <= 8:
            closest_stone = stone_distances[0][0]
            prioritized_moves = []
            for move in moves:
                if move.get('action') in ['move', 'push'] and move.get('from') == list(closest_stone):
                    success, new_board = simulate_move(board, move, self.player, rows, cols, score_cols)
                    if success:
                        to_pos = move.get('to')
                        if to_pos:
                            new_dist = _get_shortest_path_to_score((int(to_pos[0]), int(to_pos[1])),
                                                                   self.player, new_board, rows, cols, score_cols)
                            if new_dist < stone_distances[0][1]:
                                prioritized_moves.append((move, new_dist))
            if prioritized_moves:
                prioritized_moves.sort(key=lambda x: x[1])
                return prioritized_moves[0][0]
        proximity_moves = []
        for move in moves:
            if move.get('action') in ['move']:
                to_pos = move.get('to')
                if to_pos:
                    dist = _get_shortest_path_to_score((int(to_pos[0]), int(to_pos[1])),
                                                       self.player, board, rows, cols, score_cols)
                    if dist <= 5:
                        proximity_moves.append((move, dist))

        if proximity_moves:
            proximity_moves.sort(key=lambda x: x[1])
            best_prox_moves = [m for m, d in proximity_moves if d == proximity_moves[0][1]]
            if len(best_prox_moves) == 1:
                return best_prox_moves[0]

        filtered_moves = moves
        min_stones = 6
        max_rivers = 4
        if len(player_rivers) >= max_rivers or len(player_stones) <= min_stones:
            filtered_moves = [move for move in moves if not (
                    move.get('action') == 'flip' and
                    move.get('orientation') in ['horizontal', 'vertical']
            )]
            if not filtered_moves:
                filtered_moves = moves
        non_repetitive_moves = []
        if self.move_history:
            for move in filtered_moves:
                is_repetitive = self._is_repetitive_move(move)
                if not is_repetitive:
                    non_repetitive_moves.append(move)
            if len(non_repetitive_moves) > 1:
                filtered_moves = non_repetitive_moves
        best_score = float('-inf')
        if current_player_time < 5.0:
            if non_repetitive_moves and len(non_repetitive_moves) > 0:
                selected_move = random.choice(non_repetitive_moves)
            else:
                selected_move = random.choice(filtered_moves)
            self._update_move_history(selected_move)
            return selected_move
        max_depth = 1
        if current_player_time < 20.0:
            max_depth = 1
        elif current_player_time > 60.0:
            max_depth = 2
        max_same_score_moves = 5
        best_moves = []
        for move in filtered_moves:
            success, new_board = simulate_move(board, move, self.player, rows, cols, score_cols)
            if not success:
                continue
            score = self.minimax(new_board, max_depth - 1, False, rows, cols, score_cols,
                                 float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                if len(best_moves) < max_same_score_moves:
                    best_moves.append(move)
                elif random.random() < 0.2:
                    replace_idx = random.randrange(len(best_moves))
                    best_moves[replace_idx] = move
        if best_moves:
            selected_move = random.choice(best_moves)
            self._update_move_history(selected_move)
        else:
            selected_move = random.choice(filtered_moves) if filtered_moves else None
            if selected_move:
                self._update_move_history(selected_move)
        return selected_move

    def minimax(self, board: List[List[Any]], depth: int, is_maximizing: bool, rows: int, cols: int,
                score_cols: List[int], alpha=float('-inf'), beta=float('inf')) -> float:
        player_stones_in_score = count_stones_in_scoring_area(board, self.player, rows, cols, score_cols)
        opponent_stones_in_score = count_stones_in_scoring_area(board, self.opponent, rows, cols, score_cols)
        if player_stones_in_score >= 4:
            return float('inf')
        if opponent_stones_in_score >= 4:
            return float('-inf')
        if depth <= 0:
            eval_score = basic_evaluate_board(board, self.player, rows, cols, score_cols)
            return eval_score
        player = self.player if is_maximizing else self.opponent
        moves = generate_all_moves(board, player, rows, cols, score_cols)
        if not moves:
            return basic_evaluate_board(board, self.player, rows, cols, score_cols)
        if is_maximizing:
            best_score = float('-inf')
            for move in moves:
                success, new_board = simulate_move(board, move, player, rows, cols, score_cols)
                if not success:
                    continue
                score = self.minimax(new_board, depth - 1, False, rows, cols, score_cols, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in moves:
                success, new_board = simulate_move(board, move, player, rows, cols, score_cols)
                if not success:
                    continue
                score = self.minimax(new_board, depth - 1, True, rows, cols, score_cols, alpha, beta)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
