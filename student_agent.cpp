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
    static constexpr int MAX_MOVES_PER_POSITION = 35;

    // Evaluation weights
    static constexpr double WEIGHT_PIECES_IN_GOAL = 1000.0;
    static constexpr double WEIGHT_PIECES_NEAR_GOAL = 100.0;
    static constexpr double WEIGHT_ADVANCEMENT = 20.0;
    static constexpr double WEIGHT_SPREAD = 5.0;

public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()), random_double(0.0, 1.0) {
        opponent = (this->side == "circle") ? "square" : "circle";
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>>& board,
                int rows, int cols, const std::vector<int>& score_cols,
                float current_player_time, float opponent_time) {

        auto start_time = std::chrono::steady_clock::now();
        GameState state(board, side, rows, cols, score_cols);

        // Quick win check
        if (isWinning(state, side)) {
            auto moves = generateAllMoves(state, side);
            for (const auto& move : moves) {
                GameState next_state = applyMove(state, move, side);
                if (isWinning(next_state, side)) {
                    return move;
                }
            }
        }

        // Time budget
        double move_budget = std::min(MAX_MOVE_TIME, std::max(0.05, current_player_time * 0.012));

        Move best_move;
        double best_score = -INFINITY_VAL;

        // Generate and diversify moves
        auto moves = generateAllMoves(state, side);
        if (moves.empty()) {
            return {"move", {0,0}, {0,0}, {}, ""};
        }

        // Remove recent moves to prevent loops
        filterRecentMoves(moves);

        // Order moves with variety
        orderMoves(moves, state, side);

        // Limit moves but ensure variety
        if (moves.size() > MAX_MOVES_PER_POSITION) {
            // Keep top moves but add some random ones for variety
            int keep_top = MAX_MOVES_PER_POSITION * 0.7;
            int keep_random = MAX_MOVES_PER_POSITION - keep_top;

            // Shuffle the lower-scored moves and keep some
            if (moves.size() > keep_top) {
                std::shuffle(moves.begin() + keep_top, moves.end(), gen);
            }
            moves.resize(std::min((int)moves.size(), MAX_MOVES_PER_POSITION));
        }

        // Shuffle moves with similar scores to add variety
        shuffleSimilarMoves(moves);

        // Iterative deepening with randomization
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

                // Add randomization to break ties and prevent loops
                move_score += (random_double(gen) - 0.5) * 0.3; // Random -0.15 to +0.15

                if (move_score > best_score) {
                    best_score = move_score;
                    best_move = move;
                }
            }

            if (time_expired) break;
        }

        // Track this move to prevent immediate repetition
        if (!best_move.action.empty()) {
            addToHistory(best_move);
        } else {
            best_move = moves[0]; // Fallback
        }

        return best_move;
    }

private:
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

    void shuffleSimilarMoves(std::vector<Move>& moves) {
        const double SCORE_TOLERANCE = 1.0;

        for (size_t i = 0; i < moves.size(); i++) {
            for (size_t j = i + 1; j < moves.size(); j++) {
                if (std::abs(moves[i].evaluation_score - moves[j].evaluation_score) < SCORE_TOLERANCE) {
                    if (random_double(gen) > 0.5) {
                        std::swap(moves[i], moves[j]);
                    }
                }
            }
        }
    }

    void orderMoves(std::vector<Move>& moves, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;

        for (auto& move : moves) {
            double score = 0.0;

            if (move.action == "move" || move.action == "push") {
                int target_y = move.to[1];
                int source_y = move.from[1];
                int target_x = move.to[0];
                int source_x = move.from[0];

                // MAJOR FIX: Reward progress toward goal
                int old_distance = std::abs(source_y - goal_row);
                int new_distance = std::abs(target_y - goal_row);
                int progress = old_distance - new_distance; // Positive = closer to goal
                score += 50.0 * progress; // Strong reward for forward progress

                // Prefer moves in scoring columns
                for (int col : state.score_cols) {
                    if (target_x == col) {
                        score += 30.0;
                        break;
                    }
                }

                // Encourage piece spreading (avoid clustering)
                score += 3.0 * std::abs(target_x - 6); // Spread from center

                // Add variety based on piece position
                score += 2.0 * (source_x + source_y); // Position-based variety

                // Prefer pushes over simple moves
                if (move.action == "push") {
                    score += 15.0;
                }
            }

            // Encourage different types of moves for variety
            if (move.action == "flip") {
                score += 12.0 + random_double(gen) * 5.0; // Some randomness
            }
            if (move.action == "rotate") {
                score += 8.0 + random_double(gen) * 3.0;
            }

            // Add base randomization to prevent deterministic ordering
            score += random_double(gen) * 2.0;

            move.evaluation_score = score;
        }

        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.evaluation_score > b.evaluation_score;
        });
    }

    double minimax(const GameState& state, int depth, double alpha, double beta,
                   bool maximizing_player, const std::chrono::steady_clock::time_point& start_time,
                   double time_limit) {

        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            current_time - start_time).count();
        if (elapsed > time_limit) {
            return evaluatePosition(state);
        }

        if (depth == 0 || isGameOver(state)) {
            return evaluatePosition(state);
        }

        std::string current_player = maximizing_player ? side : opponent;
        auto moves = generateAllMoves(state, current_player);

        // Limit moves in deep search
        if (moves.size() > 20 && depth > 1) {
            orderMoves(moves, state, current_player);
            moves.resize(20);
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

    double evaluatePosition(const GameState& state) {
        double score = 0.0;

        // Goal-based scoring
        int my_pieces_in_goal = countPiecesInGoal(state, side);
        int opponent_pieces_in_goal = countPiecesInGoal(state, opponent);

        if (my_pieces_in_goal >= 4) return INFINITY_VAL;
        if (opponent_pieces_in_goal >= 4) return -INFINITY_VAL;

        score += WEIGHT_PIECES_IN_GOAL * (my_pieces_in_goal - opponent_pieces_in_goal);

        // Near-goal scoring
        int my_pieces_near_goal = countPiecesNearGoal(state, side);
        int opponent_pieces_near_goal = countPiecesNearGoal(state, opponent);
        score += WEIGHT_PIECES_NEAR_GOAL * (my_pieces_near_goal - opponent_pieces_near_goal);

        // Advancement scoring - NEW
        score += WEIGHT_ADVANCEMENT * evaluateAdvancement(state);

        // Piece distribution - NEW
        score += WEIGHT_SPREAD * evaluatePieceSpread(state);

        return score;
    }

    double evaluateAdvancement(const GameState& state) {
        double my_advancement = 0.0, opponent_advancement = 0.0;

        int my_goal = (side == "circle") ? 2 : state.rows - 3;
        int opponent_goal = (opponent == "circle") ? 2 : state.rows - 3;

        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty()) continue;

                if (cell.at("side") == "stone") {
                    int distance_to_goal;
                    if (cell.at("owner") == side) {
                        distance_to_goal = std::abs(y - my_goal);
                        my_advancement += 10.0 / (distance_to_goal + 1);

                        // Extra bonus for stones in scoring columns
                        for (int col : state.score_cols) {
                            if (x == col) {
                                my_advancement += 15.0 / (distance_to_goal + 1);
                                break;
                            }
                        }
                    } else if (cell.at("owner") == opponent) {
                        distance_to_goal = std::abs(y - opponent_goal);
                        opponent_advancement += 10.0 / (distance_to_goal + 1);

                        for (int col : state.score_cols) {
                            if (x == col) {
                                opponent_advancement += 15.0 / (distance_to_goal + 1);
                                break;
                            }
                        }
                    }
                }
            }
        }

        return my_advancement - opponent_advancement;
    }

    double evaluatePieceSpread(const GameState& state) {
        // Encourage spreading pieces across different columns
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

        // Calculate spread (more unique columns = better)
        std::sort(my_columns.begin(), my_columns.end());
        std::sort(opponent_columns.begin(), opponent_columns.end());

        my_columns.erase(std::unique(my_columns.begin(), my_columns.end()), my_columns.end());
        opponent_columns.erase(std::unique(opponent_columns.begin(), opponent_columns.end()), opponent_columns.end());

        return static_cast<double>(my_columns.size()) - static_cast<double>(opponent_columns.size());
    }

    // Helper functions (keeping existing ones)
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

    int countPiecesNearGoal(const GameState& state, const std::string& player) {
        int count = 0;
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
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

    std::vector<Move> generateAllMoves(const GameState& state, const std::string& player) {
        std::vector<Move> moves;
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};

        for (int y = 0; y < state.rows && moves.size() < MAX_MOVES_PER_POSITION * 2; y++) {
            for (int x = 0; x < state.cols && moves.size() < MAX_MOVES_PER_POSITION * 2; x++) {
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
                    } else if (state.board[ny][nx].at("owner") != player) {
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
                }
            }
        }

        return moves;
    }

    // Rest of helper functions remain the same...
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