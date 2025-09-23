#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <set>

namespace py = pybind11;

struct Move {
    std::string action;
    std::vector<int> from;
    std::vector<int> to;
    std::vector<int> pushed_to;
    std::string orientation;
    double evaluation_score = 0.0;
};

struct GameState {
    std::vector<std::vector<std::map<std::string, std::string>>> board;
    std::string current_player;
    int rows, cols;
    std::vector<int> score_cols;

    GameState(const std::vector<std::vector<std::map<std::string, std::string>>>& b, 
              const std::string& player, int r, int c, const std::vector<int>& sc)
        : board(b), current_player(player), rows(r), cols(c), score_cols(sc) {}
};

class StudentAgent {
private:
    std::string side;
    std::string opponent;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> random_double;

    // Move history to prevent repetition
    std::vector<Move> recent_moves;
    static constexpr int HISTORY_SIZE = 4;

    // Performance constants
    static constexpr int MAX_DEPTH = 3;
    static constexpr double INFINITY_VAL = 1000000.0;
    static constexpr double TIME_BUFFER = 0.3;
    static constexpr double MAX_MOVE_TIME = 0.6;
    static constexpr int MAX_MOVES_PER_POSITION = 50;

    // WINNING-FOCUSED evaluation weights
    static constexpr double WEIGHT_PIECES_IN_GOAL = 1000.0;      
    static constexpr double WEIGHT_PIECES_NEAR_GOAL = 120.0; // Increased     
    static constexpr double WEIGHT_ADVANCEMENT = 25.0; // Increased           
    static constexpr double WEIGHT_SPREAD = 15.0;               
    static constexpr double WEIGHT_RIVER_PUSH = 85.0;           
    static constexpr double WEIGHT_DEFENSE = 20.0;              
    static constexpr double WEIGHT_RIVER_CREATION = 12.0;       
    static constexpr double WEIGHT_BOARD_CONTROL = 10.0;        

public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()), random_double(0.0, 1.0) {
        opponent = (this->side == "circle") ? "square" : "circle";
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>>& board,
                int rows, int cols, const std::vector<int>& score_cols,
                float current_player_time, float opponent_time) {

        auto start_time = std::chrono::steady_clock::now();
        GameState state(board, side, rows, cols, score_cols);

        // URGENT: Check for immediate wins first
        auto moves = generateAllMoves(state, side);
        for (const auto& move : moves) {
            GameState next_state = applyMove(state, move, side);
            if (isWinning(next_state, side)) {
                return move; // IMMEDIATE WIN - return instantly!
            }
        }

        if (moves.empty()) {
            return {"move", {0,0}, {0,0}, {}, ""};
        }

        // Time budget
        double move_budget = std::min(MAX_MOVE_TIME, std::max(0.05, current_player_time * 0.012));

        Move best_move;
        double best_score = -INFINITY_VAL;

        // Remove recent moves to prevent loops
        filterRecentMoves(moves);

        // URGENT: Winning-focused move ordering
        orderMovesForWinning(moves, state, side);

        // Allow more moves for winning opportunities
        if (moves.size() > MAX_MOVES_PER_POSITION) {
            // Keep more top moves when looking for wins
            int keep_top = MAX_MOVES_PER_POSITION * 0.8; // Increased from 0.6
            if (moves.size() > keep_top) {
                std::shuffle(moves.begin() + keep_top, moves.end(), gen);
            }
            moves.resize(std::min((int)moves.size(), MAX_MOVES_PER_POSITION));
        }

        // Less shuffling - preserve winning move priority
        shuffleSimilarMovesLess(moves);

        // Iterative deepening with win focus
        for (int depth = 1; depth <= MAX_DEPTH; depth++) {
            bool time_expired = false;

            for (const auto& move : moves) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                    current_time - start_time).count();

                if (elapsed > move_budget) {
                    time_expired = true;
                    break;
                }

                GameState next_state = applyMove(state, move, side);
                double move_score = minimax(next_state, depth - 1, -INFINITY_VAL, INFINITY_VAL,
                                          false, start_time, move_budget);

                // Less randomization to preserve winning moves
                move_score += (random_double(gen) - 0.5) * 0.2; // Reduced from 0.5

                if (move_score > best_score) {
                    best_score = move_score;
                    best_move = move;
                }
            }

            if (time_expired) break;
        }

        // Track move history
        if (!best_move.action.empty()) {
            addToHistory(best_move);
        } else {
            best_move = moves[0];
        }

        return best_move;
    }

private:
    // Generate all moves (same as before)
    std::vector<Move> generateAllMoves(const GameState& state, const std::string& player) {
        std::vector<Move> moves;

        for (int y = 0; y < state.rows && moves.size() < MAX_MOVES_PER_POSITION * 3; y++) {
            for (int x = 0; x < state.cols && moves.size() < MAX_MOVES_PER_POSITION * 3; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty() || cell.at("owner") != player) continue;

                std::string side_type = cell.at("side");

                if (side_type == "stone") {
                    generateStoneMoves(moves, state, x, y, player);
                    moves.push_back({"flip", {x,y}, {x,y}, {}, "horizontal"});
                    moves.push_back({"flip", {x,y}, {x,y}, {}, "vertical"});

                } else if (side_type == "river") {
                    generateRiverMoves(moves, state, x, y, player);
                    generateRiverPushMoves(moves, state, x, y, player);
                    moves.push_back({"rotate", {x,y}, {x,y}, {}, ""});
                    moves.push_back({"flip", {x,y}, {x,y}, {}, ""});
                }

                if (moves.size() >= MAX_MOVES_PER_POSITION * 3) break;
            }
        }

        return moves;
    }

    // URGENT: Winning-focused move ordering
    void orderMovesForWinning(std::vector<Move>& moves, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;

        // Game phase analysis (simplified for winning focus)
        int total_pieces = countTotalPieces(state);
        int my_rivers = countMyRivers(state, player);
        int my_stones = countMyStones(state, player);
        double river_percentage = (my_stones + my_rivers > 0) ? (double)my_rivers / (my_rivers + my_stones) : 0.0;

        for (auto& move : moves) {
            double score = 0.0;

            // CRITICAL: Check for immediate winning moves FIRST
            if (isImmediateWinningMove(move, state, player)) {
                score = 10000.0; // MAXIMUM PRIORITY
                move.evaluation_score = score;
                continue;
            }

            // CRITICAL: Check for direct scoring moves
            if (isDirectScoringMove(move, state, player)) {
                score = 5000.0; // Very high priority
            }

            if (move.action == "flip") {
                int flip_x = move.from[0], flip_y = move.from[1];

                if (!state.board[flip_y][flip_x].empty() && 
                    state.board[flip_y][flip_x].at("side") == "river" &&
                    state.board[flip_y][flip_x].at("owner") == player) {

                    // CRITICAL: Converting river to stone in goal area
                    if (flip_y == goal_row) {
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += 8000.0; // HUGE priority for goal area flip
                                break;
                            }
                        }
                    }

                    // High priority for near-goal river conversions
                    int distance_to_goal = std::abs(flip_y - goal_row);
                    if (distance_to_goal <= 1) {
                        score += 2000.0; // Very high priority
                        // Bonus if in scoring column
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += 1000.0;
                                break;
                            }
                        }
                    } else if (distance_to_goal <= 2) {
                        score += 800.0; // High priority
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += 400.0;
                                break;
                            }
                        }
                    }

                } else if (!state.board[flip_y][flip_x].empty() && 
                          state.board[flip_y][flip_x].at("side") == "stone" &&
                          state.board[flip_y][flip_x].at("owner") == player) {

                    // Converting stone to river - be more conservative
                    if (river_percentage < 0.3) {
                        score += 15.0; // Moderate score

                        // Penalty for converting near goal
                        int distance_to_goal = std::abs(flip_y - goal_row);
                        if (distance_to_goal <= 3) {
                            score -= 50.0; // Strong penalty
                        }
                    } else {
                        score += 5.0; // Low score if too many rivers
                    }
                }

            } else if (move.action == "move") {
                int target_y = move.to[1];
                int source_y = move.from[1];
                int target_x = move.to[0];

                // CRITICAL: Moving to goal area
                if (target_y == goal_row) {
                    for (int col : state.score_cols) {
                        if (target_x == col) {
                            // Check if piece being moved is a stone
                            const auto& piece = state.board[source_y][move.from[0]];
                            if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "stone") {
                                score += 7000.0; // MASSIVE bonus for moving stone to goal
                            } else {
                                score += 3000.0; // High bonus for positioning in goal
                            }
                            break;
                        }
                    }
                }

                // High priority for advancing toward goal
                int old_distance = std::abs(source_y - goal_row);
                int new_distance = std::abs(target_y - goal_row);
                int progress = old_distance - new_distance;

                if (progress > 0) {
                    score += 150.0 * progress; // Much higher than before

                    // Massive bonuses for getting close to goal
                    if (new_distance == 0) {
                        score += 1000.0; // Reaching goal row
                    } else if (new_distance == 1) {
                        score += 600.0; // Adjacent to goal
                    } else if (new_distance <= 2) {
                        score += 300.0; // Very close to goal
                    }

                    // Extra bonus if moving toward scoring column
                    for (int col : state.score_cols) {
                        if (target_x == col) {
                            score += 100.0;
                            break;
                        }
                    }
                }

            } else if (move.action == "push" ) {
                int from_x = move.from[0], from_y = move.from[1];
                if(state.board[from_y][from_x].at("side") == "river"){
                    score += 120.0; // High base score
                    evaluateRiverPush(score, move, state, player);
                    continue;
                }
                // Enhanced push evaluation
                score += 50.0; // Base push value

                // Check if pushing toward goal
                int target_y = move.to[1];
                int pushed_y = move.pushed_to[1];

                if (std::abs(pushed_y - goal_row) < std::abs(target_y - goal_row)) {
                    score += 200.0; // Bonus for pushing toward our goal
                }

                // Defensive push
                int opponent_goal = (opponent == "circle") ? 2 : state.rows - 3;
                if (std::abs(pushed_y - opponent_goal) > std::abs(target_y - opponent_goal)) {
                    score += 100.0; // Bonus for pushing away from opponent goal
                }

            } else if (move.action == "rotate") {
                score += 10.0; // Lower priority
                if (improvesFlowPattern(move, state)) {
                    score += 20.0;
                }
            }

            // Minimal randomization to preserve winning moves
            score += random_double(gen) * 2.0;

            move.evaluation_score = score;
        }

        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.evaluation_score > b.evaluation_score;
        });
    }

    // NEW: Check if move immediately wins the game
    bool isImmediateWinningMove(const Move& move, const GameState& state, const std::string& player) {
        GameState next_state = applyMove(state, move, player);
        return isWinning(next_state, player);
    }

    // NEW: Check if move directly contributes to scoring
    bool isDirectScoringMove(const Move& move, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;

        if (move.action == "move" && move.to[1] == goal_row) {
            // Moving to goal row
            for (int col : state.score_cols) {
                if (move.to[0] == col) {
                    // Check if piece being moved is a stone
                    const auto& piece = state.board[move.from[1]][move.from[0]];
                    if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "stone") {
                        return true; // Direct scoring move!
                    }
                }
            }
        }

        if (move.action == "flip" && move.from[1] == goal_row) {
            // Flipping in goal row
            for (int col : state.score_cols) {
                if (move.from[0] == col) {
                    // Check if converting river to stone in goal area
                    const auto& piece = state.board[move.from[1]][move.from[0]];
                    if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "river") {
                        return true; // Direct scoring flip!
                    }
                }
            }
        }

        return false;
    }

    // CORRECTED: Include rivers in goal area (same as before)
    int countPiecesNearGoal(const GameState& state, const std::string& player) {
        int count = 0;
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        std::vector<int> check_rows = {goal_row - 1, goal_row + 1};

        // Check adjacent rows
        for (int row : check_rows) {
            if (row >= 0 && row < state.rows) {
                for (int col : state.score_cols) {
                    if (!state.board[row][col].empty() && 
                        state.board[row][col].at("owner") == player &&
                        state.board[row][col].at("side") == "stone") {
                        count++;
                    }
                }
            }
        }

        // INCLUDE rivers in goal row - these can flip to score!
        for (int col : state.score_cols) {
            if (!state.board[goal_row][col].empty() && 
                state.board[goal_row][col].at("owner") == player &&
                state.board[goal_row][col].at("side") == "river") {
                count++;
            }
        }

        return count;
    }

    // Game phase analysis functions (simplified)
    int countMyRivers(const GameState& state, const std::string& player) {
        int count = 0;
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == player && cell.at("side") == "river") {
                    count++;
                }
            }
        }
        return count;
    }

    int countMyStones(const GameState& state, const std::string& player) {
        int count = 0;
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == player && cell.at("side") == "stone") {
                    count++;
                }
            }
        }
        return count;
    }

    int countTotalPieces(const GameState& state) {
        int count = 0;
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                if (!state.board[y][x].empty()) {
                    count++;
                }
            }
        }
        return count;
    }

    // Enhanced river push evaluation (simplified for winning focus)
    void evaluateRiverPush(double& score, const Move& move, const GameState& state, const std::string& player) {
        int piece_x = move.to[0], piece_y = move.to[1];
        int final_x = move.pushed_to[0], final_y = move.pushed_to[1];

        int distance = std::abs(final_x - piece_x) + std::abs(final_y - piece_y);
        score += 5.0 * distance;

        if (!state.board[piece_y][piece_x].empty()) {
            std::string piece_owner = state.board[piece_y][piece_x].at("owner");
            int goal_row = (player == "circle") ? 2 : state.rows - 3;
            int opponent_goal = (opponent == "circle") ? 2 : state.rows - 3;

            if (piece_owner == player) {
                // Pushing our piece toward opponent goal
                int old_dist = std::abs(piece_y - opponent_goal);
                int new_dist = std::abs(final_y - opponent_goal);
                if (new_dist < old_dist) score += 40.0;

                // Bonus for scoring columns
                for (int col : state.score_cols) {
                    if (final_x == col) {
                        score += 30.0;
                        break;
                    }
                }

            } else {
                // Pushing opponent piece away from our goal
                int old_dist = std::abs(piece_y - goal_row);
                int new_dist = std::abs(final_y - goal_row);
                if (new_dist > old_dist) score += 50.0;
            }
        }
    }

    // Less shuffling to preserve winning move priorities
    void shuffleSimilarMovesLess(std::vector<Move>& moves) {
        const double SCORE_TOLERANCE = 5.0; // Increased tolerance

        // Only shuffle moves that are NOT in the top winning tier
        for (size_t i = 0; i < moves.size(); i++) {
            if (moves[i].evaluation_score >= 1000.0) continue; // Don't shuffle winning moves

            for (size_t j = i + 1; j < moves.size(); j++) {
                if (moves[j].evaluation_score >= 1000.0) continue;

                if (std::abs(moves[i].evaluation_score - moves[j].evaluation_score) < SCORE_TOLERANCE) {
                    if (random_double(gen) > 0.7) { // Less likely to shuffle
                        std::swap(moves[i], moves[j]);
                    }
                }
            }
        }
    }

    // Check if rotation improves flow patterns (simplified)
    bool improvesFlowPattern(const Move& move, const GameState& state) {
        // Simplified - assume most rotations are beneficial
        return random_double(gen) > 0.6;
    }

    // Stone moves, river moves, etc. (same as before)
    void generateStoneMoves(std::vector<Move>& moves, const GameState& state, int x, int y, 
                           const std::string& player) {
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};

        for (auto [dx, dy] : dirs) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;
            if (isOpponentScoreCell(nx, ny, player, state.rows, state.cols, state.score_cols)) continue;

            if (state.board[ny][nx].empty()) {
                moves.push_back({"move", {x,y}, {nx,ny}, {}, ""});
            } else if (state.board[ny][nx].at("owner") != player && state.board[ny][nx].at("side") == "stone") {
                int nx2 = x + 2*dx, ny2 = y + 2*dy;
                if (nx2 >= 0 && ny2 >= 0 && nx2 < state.cols && ny2 < state.rows &&
                    state.board[ny2][nx2].empty() &&
                    !isPlayerScoreCell(nx2, ny2, player, state.rows, state.cols, state.score_cols)) {
                    moves.push_back({"push", {x,y}, {nx,ny}, {nx2,ny2}, ""});
                }
            } else if (state.board[ny][nx].at("owner") == player && state.board[ny][nx].at("side") == "stone") {
                int nx2 = x + 2*dx, ny2 = y + 2*dy;
                if (nx2 >= 0 && ny2 >= 0 && nx2 < state.cols && ny2 < state.rows &&
                    state.board[ny2][nx2].empty() &&
                    !isOpponentScoreCell(nx2, ny2, player, state.rows, state.cols, state.score_cols)) {
                    moves.push_back({"push", {x,y}, {nx,ny}, {nx2,ny2}, ""});
                }
            }
        }
    }

    void generateRiverMoves(std::vector<Move>& moves, const GameState& state, int x, int y, 
                           const std::string& player) {
        std::vector<std::pair<int,int>> all_dirs = {{1,0},{-1,0},{0,1},{0,-1}};

        for (auto [dx, dy] : all_dirs) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows &&
                state.board[ny][nx].empty() &&
                !isOpponentScoreCell(nx, ny, player, state.rows, state.cols, state.score_cols)) {
                moves.push_back({"move", {x,y}, {nx,ny}, {}, ""});
            }
        }
    }

    void generateRiverPushMoves(std::vector<Move>& moves, const GameState& state, int river_x, int river_y, 
                               const std::string& player) {
        const auto& river_cell = state.board[river_y][river_x];
        std::string orientation = river_cell.at("orientation");

        std::vector<std::pair<int,int>> adjacent_dirs = {{1,0},{-1,0},{0,1},{0,-1}};

        for (auto [dx, dy] : adjacent_dirs) {
            int adj_x = river_x + dx;
            int adj_y = river_y + dy;

            if (adj_x < 0 || adj_x >= state.cols || adj_y < 0 || adj_y >= state.rows) continue;
            if (state.board[adj_y][adj_x].empty()) continue;

            const auto& adjacent_piece = state.board[adj_y][adj_x];
            if (adjacent_piece.at("side") != "stone") continue;

            std::vector<std::pair<int, int>> flow_dirs;
            if (orientation == "horizontal") {
                flow_dirs = {{1,0}, {-1,0}};
            } else {
                flow_dirs = {{0,1}, {0,-1}};
            }

            for (auto [flow_dx, flow_dy] : flow_dirs) {
                auto final_pos = traceRiverPushPath(state, adj_x, adj_y, flow_dx, flow_dy);

                if (final_pos.first != adj_x || final_pos.second != adj_y) {
                    moves.push_back({"push", {river_x, river_y}, {adj_x, adj_y}, 
                                   {final_pos.first, final_pos.second}, orientation});
                }
            }
        }
    }

    std::pair<int, int> traceRiverPushPath(const GameState& state, int start_x, int start_y, 
                                          int flow_dx, int flow_dy) {
        int current_x = start_x;
        int current_y = start_y;
        int current_flow_dx = flow_dx;
        int current_flow_dy = flow_dy;

        std::set<std::pair<int, int>> visited;

        while (true) {
            int next_x = current_x + current_flow_dx;
            int next_y = current_y + current_flow_dy;

            if (next_x < 0 || next_x >= state.cols || next_y < 0 || next_y >= state.rows) break;
            if (visited.count({next_x, next_y})) break;
            if (isOpponentScoreCell(next_x, next_y, state.current_player, state.rows, state.cols, state.score_cols)) break;

            if (state.board[next_y][next_x].empty()) {
                current_x = next_x;
                current_y = next_y;
                visited.insert({current_x, current_y});

            } else if (state.board[next_y][next_x].at("side") == "stone") {
                break;

            } else if (state.board[next_y][next_x].at("side") == "river") {
                const auto& hit_river = state.board[next_y][next_x];
                std::string hit_orientation = hit_river.at("orientation");

                current_x = next_x;
                current_y = next_y;
                visited.insert({current_x, current_y});

                std::vector<std::pair<int, int>> new_flows;
                if (hit_orientation == "horizontal") {
                    new_flows = {{1,0}, {-1,0}};
                } else {
                    new_flows = {{0,1}, {0,-1}};
                }

                bool found_direction = false;
                for (auto [new_dx, new_dy] : new_flows) {
                    if (new_dx != -current_flow_dx || new_dy != -current_flow_dy) {
                        current_flow_dx = new_dx;
                        current_flow_dy = new_dy;
                        found_direction = true;
                        break;
                    }
                }

                if (!found_direction) break;
            }
        }

        return {current_x, current_y};
    }

    GameState applyMove(const GameState& state, const Move& move, const std::string& player) {
        GameState new_state = state;

        if (move.action == "move") {
            int fx = move.from[0], fy = move.from[1];
            int tx = move.to[0], ty = move.to[1];
            new_state.board[ty][tx] = state.board[fy][fx];
            new_state.board[fy][fx].clear();

        } else if (move.action == "push") {
            int fx = move.from[0], fy = move.from[1];
            int tx = move.to[0], ty = move.to[1];
            int px = move.pushed_to[0], py = move.pushed_to[1];
            new_state.board[py][px] = state.board[ty][tx];
            new_state.board[ty][tx] = state.board[fy][fx];
            new_state.board[fy][fx].clear();
            auto mover = new_state.board[ty][tx];
            if(mover.at("side") == "river"){
                new_state.board[ty][tx]["side"] = "stone";
                new_state.board[ty][tx].erase("orientation");
            }

        } else if (move.action == "flip") {
            int fx = move.from[0], fy = move.from[1];
            if (state.board[fy][fx].at("side") == "stone") {
                new_state.board[fy][fx]["side"] = "river";
                new_state.board[fy][fx]["orientation"] = move.orientation;
            } else {
                new_state.board[fy][fx]["side"] = "stone";
                new_state.board[fy][fx].erase("orientation");
            }

        } else if (move.action == "rotate") {
            int fx = move.from[0], fy = move.from[1];
            std::string current_orientation = state.board[fy][fx].at("orientation");
            new_state.board[fy][fx]["orientation"] = (current_orientation == "horizontal") ? "vertical" : "horizontal";
        }

        return new_state;
    }

    // Helper functions
    void filterRecentMoves(std::vector<Move>& moves) {
        moves.erase(
            std::remove_if(moves.begin(), moves.end(), [this](const Move& move) {
                return isMoveRecent(move);
            }),
            moves.end()
        );
    }

    bool isMoveRecent(const Move& move) {
        for (const auto& recent : recent_moves) {
            if (recent.action == move.action && 
                recent.from == move.from && 
                recent.to == move.to) {
                return true;
            }
        }
        return false;
    }

    void addToHistory(const Move& move) {
        recent_moves.push_back(move);
        if (recent_moves.size() > HISTORY_SIZE) {
            recent_moves.erase(recent_moves.begin());
        }
    }

    double minimax(const GameState& state, int depth, double alpha, double beta,
                   bool maximizing_player, const std::chrono::steady_clock::time_point& start_time,
                   double time_limit) {

        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            current_time - start_time).count();
        if (elapsed > time_limit) {
            return evaluateForWinning(state);
        }

        if (depth == 0 || isGameOver(state)) {
            return evaluateForWinning(state);
        }

        std::string current_player = maximizing_player ? side : opponent;
        auto moves = generateAllMoves(state, current_player);

        if (moves.size() > 30 && depth > 1) {
            orderMovesForWinning(moves, state, current_player);
            moves.resize(30);
        }

        if (maximizing_player) {
            double max_eval = -INFINITY_VAL;
            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, current_player);
                double eval = minimax(next_state, depth - 1, alpha, beta, false, start_time, time_limit);
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break;
            }
            return max_eval;
        } else {
            double min_eval = INFINITY_VAL;
            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, current_player);
                double eval = minimax(next_state, depth - 1, alpha, beta, true, start_time, time_limit);
                min_eval = std::min(min_eval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) break;
            }
            return min_eval;
        }
    }

    // WINNING-FOCUSED position evaluation
    double evaluateForWinning(const GameState& state) {
        double score = 0.0;

        // Core scoring (highest priority)
        int my_pieces_in_goal = countPiecesInGoal(state, side);
        int opponent_pieces_in_goal = countPiecesInGoal(state, opponent);

        if (my_pieces_in_goal >= 4) return INFINITY_VAL;
        if (opponent_pieces_in_goal >= 4) return -INFINITY_VAL;

        score += WEIGHT_PIECES_IN_GOAL * (my_pieces_in_goal - opponent_pieces_in_goal);

        // Near-goal pieces (includes rivers in goal area!)
        int my_pieces_near_goal = countPiecesNearGoal(state, side);
        int opponent_pieces_near_goal = countPiecesNearGoal(state, opponent);
        score += WEIGHT_PIECES_NEAR_GOAL * (my_pieces_near_goal - opponent_pieces_near_goal);

        // Advancement (increased weight)
        score += WEIGHT_ADVANCEMENT * evaluateAdvancement(state);

        // Other factors (reduced weights)
        score += WEIGHT_SPREAD * evaluatePieceSpread(state);
        score += WEIGHT_DEFENSE * evaluateDefense(state);
        score += WEIGHT_BOARD_CONTROL * evaluateBoardControl(state);

        return score;
    }

    // Evaluation functions (simplified for winning focus)
    double evaluateAdvancement(const GameState& state) {
        double my_advancement = 0.0, opponent_advancement = 0.0;

        int my_goal = (side == "circle") ? 2 : state.rows - 3;
        int opponent_goal = (opponent == "circle") ? 2 : state.rows - 3;

        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty() || cell.at("side") != "stone") continue;

                int distance_to_goal;
                if (cell.at("owner") == side) {
                    distance_to_goal = std::abs(y - my_goal);
                    my_advancement += 8.0 / (distance_to_goal + 1); // Increased from 4.0

                    for (int col : state.score_cols) {
                        if (x == col) {
                            my_advancement += 15.0 / (distance_to_goal + 1); // Increased from 6.0
                            break;
                        }
                    }
                } else if (cell.at("owner") == opponent) {
                    distance_to_goal = std::abs(y - opponent_goal);
                    opponent_advancement += 8.0 / (distance_to_goal + 1);

                    for (int col : state.score_cols) {
                        if (x == col) {
                            opponent_advancement += 15.0 / (distance_to_goal + 1);
                            break;
                        }
                    }
                }
            }
        }

        return my_advancement - opponent_advancement;
    }

    double evaluateDefense(const GameState& state) {
        double defense_score = 0.0;
        int our_goal_row = (side == "circle") ? 2 : state.rows - 3;

        for (int y = our_goal_row - 2; y <= our_goal_row + 2; y++) {
            if (y < 0 || y >= state.rows) continue;
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == side) {
                    double distance_to_goal = std::abs(y - our_goal_row);
                    defense_score += 2.0 / (distance_to_goal + 1);
                }
            }
        }

        return defense_score;
    }

    double evaluateBoardControl(const GameState& state) {
        double control_score = 0.0;
        int center_x = state.cols / 2;
        int center_y = state.rows / 2;

        for (int y = center_y - 1; y <= center_y + 1; y++) {
            if (y < 0 || y >= state.rows) continue;
            for (int x = center_x - 1; x <= center_x + 1; x++) {
                if (x < 0 || x >= state.cols) continue;
                const auto& cell = state.board[y][x];
                if (!cell.empty()) {
                    if (cell.at("owner") == side) {
                        control_score += 1.0;
                    } else {
                        control_score -= 0.5;
                    }
                }
            }
        }

        return control_score;
    }

    double evaluatePieceSpread(const GameState& state) {
        std::vector<int> my_columns, opponent_columns;

        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty()) continue;

                if (cell.at("owner") == side) {
                    my_columns.push_back(x);
                } else if (cell.at("owner") == opponent) {
                    opponent_columns.push_back(x);
                }
            }
        }

        std::sort(my_columns.begin(), my_columns.end());
        std::sort(opponent_columns.begin(), opponent_columns.end());

        my_columns.erase(std::unique(my_columns.begin(), my_columns.end()), my_columns.end());
        opponent_columns.erase(std::unique(opponent_columns.begin(), opponent_columns.end()), opponent_columns.end());

        return static_cast<double>(my_columns.size()) - static_cast<double>(opponent_columns.size());
    }

    int countPiecesInGoal(const GameState& state, const std::string& player) {
        int count = 0;
        int goal_row = (player == "circle") ? 2 : state.rows - 3;

        for (int col : state.score_cols) {
            if (!state.board[goal_row][col].empty() && 
                state.board[goal_row][col].at("owner") == player &&
                state.board[goal_row][col].at("side") == "stone") {
                count++;
            }
        }
        return count;
    }

    bool isWinning(const GameState& state, const std::string& player) {
        return countPiecesInGoal(state, player) >= 4;
    }

    bool isGameOver(const GameState& state) {
        return isWinning(state, side) || isWinning(state, opponent);
    }

    bool isOpponentScoreCell(int x, int y, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
        bool in_score_col = false;
        for (int col : score_cols) {
            if (x == col) {
                in_score_col = true;
                break;
            }
        }
        if (!in_score_col) return false;

        if (player == "circle") {
            return y == (rows - 3);
        } else {
            return y == 2;
        }
    }

    bool isPlayerScoreCell(int x, int y, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
        bool in_score_col = false;
        for (int col : score_cols) {
            if (x == col) {
                in_score_col = true;
                break;
            }
        }
        if (!in_score_col) return false;

        if (player == "circle") {
            return y == 2;
        } else {
            return y == (rows - 3);
        }
    }
};

PYBIND11_MODULE(student_agent_module, m) {
    py::class_<Move>(m, "Move")
        .def_readonly("action", &Move::action)
        .def_readonly("from_pos", &Move::from)
        .def_readonly("to_pos", &Move::to)
        .def_readonly("pushed_to", &Move::pushed_to)
        .def_readonly("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose);
}