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

namespace py = pybind11;

/*
=========================================================
 ENHANCED STUDENT AGENT FOR STONES & RIVERS GAME
---------------------------------------------------------
 This implementation uses advanced game AI techniques:
 - Minimax algorithm with alpha-beta pruning
 - Iterative deepening for time management
 - Strategic evaluation function
 - Move ordering for better pruning efficiency
=========================================================
*/

// ---- Move struct ----
struct Move {
    std::string action;
    std::vector<int> from;
    std::vector<int> to;
    std::vector<int> pushed_to;
    std::string orientation;
    double evaluation_score = 0.0;  // For move ordering
};

// ---- Game State representation ----
struct GameState {
    std::vector<std::vector<std::map<std::string, std::string>>> board;
    std::string current_player;
    int rows, cols;
    std::vector<int> score_cols;

    GameState(const std::vector<std::vector<std::map<std::string, std::string>>>& b, 
              const std::string& player, int r, int c, const std::vector<int>& sc)
        : board(b), current_player(player), rows(r), cols(c), score_cols(sc) {}
};

// ---- Enhanced Student Agent ----
class StudentAgent {
private:
    std::string side;
    std::string opponent;
    std::random_device rd;
    std::mt19937 gen;

    // Game constants
    static constexpr int MAX_DEPTH = 6;
    static constexpr double INFINITY_VAL = 1000000.0;
    static constexpr double TIME_BUFFER = 0.1; // Reserve 100ms for safety

    // Evaluation weights
    static constexpr double WEIGHT_PIECES_IN_GOAL = 1000.0;
    static constexpr double WEIGHT_PIECES_NEAR_GOAL = 100.0;
    static constexpr double WEIGHT_MOBILITY = 10.0;
    static constexpr double WEIGHT_RIVER_CONTROL = 50.0;
    static constexpr double WEIGHT_DEFENSIVE = 20.0;

public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()) {
        opponent = (this->side == "circle") ? "square" : "circle";
    }

    // Main decision function
    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>>& board, 
                int rows, int cols, const std::vector<int>& score_cols, 
                float current_player_time, float opponent_time) {

        auto start_time = std::chrono::steady_clock::now();
        GameState state(board, side, rows, cols, score_cols);

        // Check if we're in a winning/losing position immediately
        if (isWinning(state, side)) {
            // If we can win, find the winning move quickly
            auto moves = generateAllMoves(state, side);
            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, side);
                if (isWinning(next_state, side)) {
                    return move;
                }
            }
        }

        // Use iterative deepening with time management
        Move best_move;
        double best_score = -INFINITY_VAL;
        int max_search_depth = std::min(MAX_DEPTH, 4); // Conservative start

        // Adjust search depth based on available time
        if (current_player_time > 30.0) max_search_depth = 5;
        if (current_player_time > 45.0) max_search_depth = 6;

        // Iterative deepening
        for (int depth = 1; depth <= max_search_depth; depth++) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                current_time - start_time).count();

            // Check if we have enough time for this depth
            if (elapsed + TIME_BUFFER > current_player_time) break;

            double score = minimax(state, depth, -INFINITY_VAL, INFINITY_VAL, true, start_time, current_player_time);

            // Update best move if we completed this depth
            auto moves = generateAllMoves(state, side);
            orderMoves(moves, state, side);

            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, side);
                double move_score = minimax(next_state, depth - 1, -INFINITY_VAL, INFINITY_VAL, false, start_time, current_player_time);

                if (move_score > best_score) {
                    best_score = move_score;
                    best_move = move;
                }
            }
        }

        // Fallback to basic move if search failed
        if (best_move.action.empty()) {
            auto moves = generateAllMoves(state, side);
            if (!moves.empty()) {
                orderMoves(moves, state, side);
                best_move = moves[0];
            } else {
                return {"move", {0,0}, {0,0}, {}, ""}; // Emergency fallback
            }
        }

        return best_move;
    }

private:
    // Minimax with alpha-beta pruning
    double minimax(const GameState& state, int depth, double alpha, double beta, 
                   bool maximizing_player, const std::chrono::steady_clock::time_point& start_time, 
                   double time_limit) {

        // Time check
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            current_time - start_time).count();
        if (elapsed + TIME_BUFFER > time_limit) {
            return evaluatePosition(state, maximizing_player ? side : opponent);
        }

        // Terminal conditions
        if (depth == 0 || isGameOver(state)) {
            return evaluatePosition(state, maximizing_player ? side : opponent);
        }

        std::string current_player = maximizing_player ? side : opponent;
        auto moves = generateAllMoves(state, current_player);
        orderMoves(moves, state, current_player);

        if (maximizing_player) {
            double max_eval = -INFINITY_VAL;
            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, current_player);
                double eval = minimax(next_state, depth - 1, alpha, beta, false, start_time, time_limit);
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break; // Alpha-beta pruning
            }
            return max_eval;
        } else {
            double min_eval = INFINITY_VAL;
            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, current_player);
                double eval = minimax(next_state, depth - 1, alpha, beta, true, start_time, time_limit);
                min_eval = std::min(min_eval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) break; // Alpha-beta pruning
            }
            return min_eval;
        }
    }

    // Enhanced evaluation function
    double evaluatePosition(const GameState& state, const std::string& player) {
        double score = 0.0;

        // Count pieces in scoring area
        int my_pieces_in_goal = countPiecesInGoal(state, side);
        int opponent_pieces_in_goal = countPiecesInGoal(state, opponent);

        // Winning condition
        if (my_pieces_in_goal >= 4) return INFINITY_VAL;
        if (opponent_pieces_in_goal >= 4) return -INFINITY_VAL;

        // Score based on pieces in goal
        score += WEIGHT_PIECES_IN_GOAL * (my_pieces_in_goal - opponent_pieces_in_goal);

        // Score based on pieces close to goal
        int my_pieces_near_goal = countPiecesNearGoal(state, side);
        int opponent_pieces_near_goal = countPiecesNearGoal(state, opponent);
        score += WEIGHT_PIECES_NEAR_GOAL * (my_pieces_near_goal - opponent_pieces_near_goal);

        // Mobility score
        int my_mobility = countValidMoves(state, side);
        int opponent_mobility = countValidMoves(state, opponent);
        score += WEIGHT_MOBILITY * (my_mobility - opponent_mobility);

        // River control score
        double my_river_control = evaluateRiverControl(state, side);
        double opponent_river_control = evaluateRiverControl(state, opponent);
        score += WEIGHT_RIVER_CONTROL * (my_river_control - opponent_river_control);

        // Defensive considerations
        double defensive_score = evaluateDefensivePosition(state, side);
        score += WEIGHT_DEFENSIVE * defensive_score;

        return score;
    }

    // Helper evaluation functions
    int countPiecesInGoal(const GameState& state, const std::string& player) {
        int count = 0;
        int goal_row = (player == "circle") ? getTopScoreRow() : getBottomScoreRow(state.rows);

        for (int col : state.score_cols) {
            if (!state.board[goal_row][col].empty() && 
                state.board[goal_row][col].at("owner") == player &&
                state.board[goal_row][col].at("side") == "stone") {
                count++;
            }
        }
        return count;
    }

    int countPiecesNearGoal(const GameState& state, const std::string& player) {
        int count = 0;
        int goal_row = (player == "circle") ? getTopScoreRow() : getBottomScoreRow(state.rows);

        // Check rows adjacent to goal
        std::vector<int> check_rows = {goal_row - 1, goal_row + 1};

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
        return count;
    }

    int countValidMoves(const GameState& state, const std::string& player) {
        return generateAllMoves(state, player).size();
    }

    double evaluateRiverControl(const GameState& state, const std::string& player) {
        double control = 0.0;

        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == player && cell.at("side") == "river") {
                    // Rivers are valuable, especially in central positions
                    double position_value = 1.0;
                    if (x >= state.cols/3 && x <= 2*state.cols/3) position_value *= 1.5;
                    if (y >= state.rows/3 && y <= 2*state.rows/3) position_value *= 1.5;
                    control += position_value;
                }
            }
        }
        return control;
    }

    double evaluateDefensivePosition(const GameState& state, const std::string& player) {
        // Evaluate how well we're defending against opponent advances
        int opponent_goal_row = (opponent == "circle") ? getTopScoreRow() : getBottomScoreRow(state.rows);
        double defensive_score = 0.0;

        // Check if we have pieces blocking opponent's path
        for (int col : state.score_cols) {
            for (int row = 0; row < state.rows; row++) {
                if (!state.board[row][col].empty() && 
                    state.board[row][col].at("owner") == player) {
                    // Bonus for pieces that can block opponent
                    int distance_to_opponent_goal = std::abs(row - opponent_goal_row);
                    if (distance_to_opponent_goal < 3) {
                        defensive_score += 1.0 / (distance_to_opponent_goal + 1);
                    }
                }
            }
        }
        return defensive_score;
    }

    // Move generation and utilities
    std::vector<Move> generateAllMoves(const GameState& state, const std::string& player) {
        std::vector<Move> moves;
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};

        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty() || cell.at("owner") != player) continue;

                std::string side_type = cell.at("side");

                // Generate moves
                for (auto [dx, dy] : dirs) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;
                    if (isOpponentScoreCell(nx, ny, player, state.rows, state.cols, state.score_cols)) continue;

                    if (state.board[ny][nx].empty()) {
                        moves.push_back({"move", {x,y}, {nx,ny}, {}, ""});
                    } else {
                        // Generate push moves
                        int nx2 = x + 2*dx, ny2 = y + 2*dy;
                        if (nx2 >= 0 && ny2 >= 0 && nx2 < state.cols && ny2 < state.rows &&
                            state.board[ny2][nx2].empty() &&
                            !isOpponentScoreCell(nx2, ny2, player, state.rows, state.cols, state.score_cols)) {
                            moves.push_back({"push", {x,y}, {nx,ny}, {nx2,ny2}, ""});
                        }
                    }
                }

                // Generate flip/rotate moves
                if (side_type == "stone") {
                    moves.push_back({"flip", {x,y}, {x,y}, {}, "horizontal"});
                    moves.push_back({"flip", {x,y}, {x,y}, {}, "vertical"});
                } else if (side_type == "river") {
                    moves.push_back({"rotate", {x,y}, {x,y}, {}, ""});
                    moves.push_back({"flip", {x,y}, {x,y}, {}, ""});
                }
            }
        }

        return moves;
    }

    // Move ordering for better alpha-beta pruning
    void orderMoves(std::vector<Move>& moves, const GameState& state, const std::string& player) {
        // Simple move ordering: prioritize moves toward goal
        for (auto& move : moves) {
            double score = 0.0;

            if (move.action == "move" || move.action == "push") {
                int target_y = move.to[1];
                int goal_row = (player == "circle") ? getTopScoreRow() : getBottomScoreRow(state.rows);

                // Prefer moves toward goal
                score += 10.0 / (std::abs(target_y - goal_row) + 1);

                // Prefer moves toward scoring columns
                int target_x = move.to[0];
                bool in_score_col = false;
                for (int col : state.score_cols) {
                    if (target_x == col) {
                        in_score_col = true;
                        break;
                    }
                }
                if (in_score_col) score += 5.0;
            }

            move.evaluation_score = score;
        }

        // Sort moves by evaluation score (highest first)
        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.evaluation_score > b.evaluation_score;
        });
    }

    // Game state manipulation
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

            new_state.board[py][px] = state.board[ty][tx]; // Move pushed piece
            new_state.board[ty][tx] = state.board[fy][fx]; // Move our piece
            new_state.board[fy][fx].clear(); // Clear original position

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

    // Game state queries
    bool isWinning(const GameState& state, const std::string& player) {
        return countPiecesInGoal(state, player) >= 4;
    }

    bool isGameOver(const GameState& state) {
        return isWinning(state, side) || isWinning(state, opponent);
    }

    // Utility functions
    int getTopScoreRow() { return 2; }
    int getBottomScoreRow(int rows) { return rows - 3; }

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
            return y == getBottomScoreRow(rows);
        } else {
            return y == getTopScoreRow();
        }
    }
};

// ---- PyBind11 bindings ----
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