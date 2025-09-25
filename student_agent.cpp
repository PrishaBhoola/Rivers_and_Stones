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
#include <queue>
#include <utility>

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
    int move_count = 0;
    std::pair<int, int> stab1 = {-1, -1}, stab2 = {-1, -1};
    std::vector<Move> recent_moves;
    static constexpr int HISTORY_SIZE = 4;
    static constexpr int EARLY_GAME_MOVES = 20; 
    static constexpr int MAX_DEPTH = 3;
    static constexpr double INFINITY_VAL = 1000000.0;
    static constexpr double MAX_MOVE_TIME = 0.6;
    static constexpr double WEIGHT_PIECES_IN_GOAL = 10000.0;
    static constexpr double WEIGHT_SCORING_THREATS = 1200.0;
    static constexpr double WEIGHT_ADVANCEMENT = 25.0;
    static constexpr double WEIGHT_SPREAD = 15.0;
    static constexpr double WEIGHT_DEFENSE = 20.0;
    static constexpr double WEIGHT_BOARD_CONTROL = 10.0;
    static constexpr double WEIGHT_HIGHWAY_NETWORK = 90.0;

public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()), random_double(0.0, 1.0) {
        opponent = (this->side == "circle") ? "square" : "circle";
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>>& board,
                int rows, int cols, const std::vector<int>& score_cols,
                float current_player_time, float opponent_time) {

        auto start_time = std::chrono::steady_clock::now();
        GameState state(board, side, rows, cols, score_cols);
        
        if (move_count < 4) {
            if (side == "circle") {
                int row = state.rows - 3;
                int col1 = score_cols[0] - 1;
                int col2 = score_cols.back() + 1;
                stab1 = {col1, row};
                stab2 = {col2, row};
                if (move_count == 0) {
                    move_count++; return {"move", {col1, row - 1}, {col1, row}, {}, ""};
                } else if (move_count == 1) {
                    move_count++; return {"move", {col2, row - 1}, {col2, row}, {}, ""};
                } else if (move_count == 2) {
                    move_count++; return {"flip", {col1, row}, {}, {}, "vertical"};
                } else if (move_count == 3) {
                    move_count++; return {"flip", {col2, row}, {}, {}, "vertical"};
                }
            } else { 
                int row = 2;
                int col1 = score_cols[0] - 1;
                int col2 = score_cols.back() + 1;
                stab1 = {col1, row};
                stab2 = {col2, row};
                if (move_count == 0) {
                    move_count++; return {"move", {col1, row + 1}, {col1, row}, {}, ""};
                } else if (move_count == 1) {
                    move_count++; return {"move", {col2, row + 1}, {col2, row}, {}, ""};
                } else if (move_count == 2) {
                    move_count++; return {"flip", {col1, row}, {}, {}, "vertical"};
                } else if (move_count == 3) {
                    move_count++; return {"flip", {col2, row}, {}, {}, "vertical"};
                }
            }
        }
        move_count++;

        auto moves = generateAllMoves(state, side);
        for (const auto& move : moves) {
            if (isWinning(applyMove(state, move, side), side)) {
                return move;
            }
        }

        if (moves.empty()) {
            return {"move", {0,0}, {0,0}, {}, ""};
        }

        double move_budget = std::min(MAX_MOVE_TIME, std::max(0.05, current_player_time * 0.015));
        Move best_move;
        double best_score = -INFINITY_VAL;

        filterRecentMoves(moves);
        orderMovesForWinning(moves, state, side);

        for (int depth = 1; depth <= MAX_DEPTH; depth++) {
            bool time_expired = false;
            for (const auto& move : moves) {
                auto current_time = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time).count() > move_budget) {
                    time_expired = true;
                    break;
                }
                GameState next_state = applyMove(state, move, side);
                double move_score = minimax(next_state, depth - 1, -INFINITY_VAL, INFINITY_VAL, false, start_time, move_budget);
                if (move_score > best_score) {
                    best_score = move_score;
                    best_move = move;
                }
            }
            if (time_expired) break;
        }

        if (!best_move.action.empty()) {
            addToHistory(best_move);
        } else if (!moves.empty()){
            best_move = moves[0];
        }

        return best_move;
    }

private:
    std::vector<Move> generateAllMoves(const GameState& state, const std::string& player) {
        std::vector<Move> moves;
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty() || cell.at("owner") != player) continue;
                
                bool is_stable_piece = (std::make_pair(x, y) == stab1 || std::make_pair(x, y) == stab2);

                if (cell.at("side") == "stone") {
                    generateStoneMoves(moves, state, x, y, player, is_stable_piece);
                    if (!is_stable_piece) {
                        moves.push_back({"flip", {x, y}, {}, {}, "horizontal"});
                        moves.push_back({"flip", {x, y}, {}, {}, "vertical"});
                    }
                } else if (cell.at("side") == "river") {
                    generateRiverMoves(moves, state, x, y, player, is_stable_piece);
                    if (!is_stable_piece) {
                         moves.push_back({"rotate", {x,y}, {x,y}, {}, ""});
                         moves.push_back({"flip", {x,y}, {x,y}, {}, ""});
                    }
                }
            }
        }
        return moves;
    }
    
    void addMoveIfAllowed(std::vector<Move>& moves, const Move& move, bool is_stable) {
        if (!is_stable) {
            moves.push_back(move);
        }
    }

    bool isOpponentScoreCell(int x, int y, const GameState& state, const std::string& player) {
        int opponent_goal_row = (player == "circle") ?  state.rows - 3 : 2;
        for (int col : state.score_cols) {
            if (x == col && y == opponent_goal_row) {
                return true;
            }
        }
        return false;
    }
    
    void generateStoneMoves(std::vector<Move>& moves, const GameState& state, int x, int y, const std::string& player, bool is_stable) {
        for (auto [dx, dy] : std::vector<std::pair<int,int>>{{1,0},{-1,0},{0,1},{0,-1}}) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;
            if(isOpponentScoreCell(nx, ny, state, player)) continue;

            if (state.board[ny][nx].empty()) {
                addMoveIfAllowed(moves, {"move", {x,y}, {nx,ny}, {}, ""}, is_stable);
            } else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("side") == "river") {
                auto destinations = traceAllRiverDestinations(state, nx, ny, x, y, player, player);
                for (auto [dest_x, dest_y] : destinations) {
                    addMoveIfAllowed(moves, {"move", {x,y}, {dest_x,dest_y}, {}, ""}, is_stable);
                }
            } else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("side") == "stone") {
                int nx2 = x + 2*dx, ny2 = y + 2*dy;
                std::string pushed_owner = state.board[ny][nx].at("owner");
                if (nx2 >= 0 && nx2 < state.cols && ny2 >= 0 && ny2 < state.rows && state.board[ny2][nx2].empty() &&
                    canPushStoneTo(nx2, ny2, pushed_owner, state.rows, state.cols, state.score_cols, player)) {
                    addMoveIfAllowed(moves, {"push", {x,y}, {nx,ny}, {nx2,ny2}, ""}, is_stable);
                }
            }
        }
    }

    void generateRiverMoves(std::vector<Move>& moves, const GameState& state, int x, int y, const std::string& player, bool is_stable) {
        generateRiverPushMoves(moves, state, x, y, player);

        for (auto [dx, dy] : std::vector<std::pair<int,int>>{{1,0},{-1,0},{0,1},{0,-1}}) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;
            if(isOpponentScoreCell(nx, ny, state, player)) continue;
            if (state.board[ny][nx].empty()) {
                addMoveIfAllowed(moves, {"move", {x, y}, {nx, ny}, {}, ""}, is_stable);
            } else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("side") == "river") {
                auto destinations = traceAllRiverDestinations(state, nx, ny, x, y, player, player);
                for (auto [dest_x, dest_y] : destinations) {
                    addMoveIfAllowed(moves, {"move", {x,y}, {dest_x,dest_y}, {}, ""}, is_stable);
                }
            }
        }
    }
    
    void generateRiverPushMoves(std::vector<Move>& moves, const GameState& state, 
                               int river_x, int river_y, const std::string& player) {
        const auto& river_cell = state.board[river_y][river_x];
        for (auto [dx, dy] : std::vector<std::pair<int,int>>{{1,0},{-1,0},{0,1},{0,-1}}) {
            int adj_x = river_x + dx, adj_y = river_y + dy;
            if (adj_x < 0 || adj_x >= state.cols || adj_y < 0 || adj_y >= state.rows || state.board[adj_y][adj_x].empty()) continue;
            if(isOpponentScoreCell(adj_x, adj_y, state, player)) continue;
            const auto& adjacent_piece = state.board[adj_y][adj_x];
            if (adjacent_piece.at("side") != "stone") continue;

            auto destinations = traceAllRiverDestinations(state, adj_x, adj_y, river_x, river_y, adjacent_piece.at("owner"), player, true);

            for (auto [dest_x, dest_y] : destinations) {
                if (dest_x != river_x || dest_y != river_y) {
                    moves.push_back({"push", {river_x, river_y}, {adj_x, adj_y}, {dest_x, dest_y}, river_cell.at("orientation")});
                }
            }
        }
    }

    std::vector<std::pair<int,int>> traceAllRiverDestinations(const GameState& state, 
                                                             int start_x, int start_y,int ox, int oy, 
                                                             const std::string& pushed_stone_owner, const std::string& player,
                                                             bool is_river_push = false) {
        std::vector<std::pair<int,int>> destinations;
        std::set<std::pair<int,int>> visited;
        std::queue<std::pair<int,int>> q;
        q.push({start_x, start_y});

        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();

            if (visited.count({x, y}) || x < 0 || x >= state.cols || y < 0 || y >= state.rows) continue;
            visited.insert({x, y});
            const auto& cell = state.board[y][x];

            if (cell.empty()) {
                if (canPushStoneTo(x, y, pushed_stone_owner, state.rows, state.cols, state.score_cols, player)) {
                    destinations.push_back({x, y});
                }
                continue;
            }
            if (cell.at("side") == "stone") {
                 if(is_river_push && x==start_x && y==start_y) {
                    const auto& cell_orig = state.board[oy][ox];
                    std::string orientation = cell_orig.at("orientation");
                    std::vector<std::pair<int,int>> flow_dirs = (orientation == "horizontal") ? 
                                                                std::vector<std::pair<int, int>>{{1,0}, {-1,0}} : std::vector<std::pair<int, int>>{{0,1}, {0,-1}};
                    for (auto [dx, dy] : flow_dirs) {
                        int nx = x + dx, ny = y + dy;
                        while (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                            if (visited.count({nx, ny})) break;
                            const auto& next_cell = state.board[ny][nx];
                            if (next_cell.empty()) {
                                if (canPushStoneTo(nx, ny, pushed_stone_owner, state.rows, state.cols, state.score_cols, player)) destinations.push_back({nx, ny});
                                nx += dx; ny += dy;
                            } else if (nx==ox && ny==oy) { nx += dx; ny += dy;
                            } else if (next_cell.at("side") == "stone") break;
                            else if (next_cell.at("side") == "river") {
                                if (!visited.count({nx, ny})) q.push({nx, ny});
                                break;
                            } else break;
                        }
                    }
                }
                continue;
            }
            if (cell.at("side") == "river") {
                std::string orientation = cell.at("orientation");
                std::vector<std::pair<int,int>> flow_dirs = (orientation == "horizontal") ? 
                                                            std::vector<std::pair<int, int>>{{1,0}, {-1,0}} : std::vector<std::pair<int, int>>{{0,1}, {0,-1}};
                for (auto [dx, dy] : flow_dirs) {
                    int nx = x + dx, ny = y + dy;
                    while (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                        if (visited.count({nx, ny})) break;
                        const auto& next_cell = state.board[ny][nx];
                        if (next_cell.empty()) {
                            if (canPushStoneTo(nx, ny, pushed_stone_owner, state.rows, state.cols, state.score_cols, player)) destinations.push_back({nx, ny});
                            nx += dx; ny += dy;
                        } else if (nx==ox && ny==oy) { nx += dx; ny += dy;
                        } else if (next_cell.at("side") == "stone") break;
                        else if (next_cell.at("side") == "river") {
                            if (!visited.count({nx, ny})) q.push({nx, ny});
                            break;
                        } else break;
                    }
                }
            }
        }
        return destinations;
    }

    bool canPushStoneTo(int x, int y, const std::string& pushed_stone_owner, 
                       int rows, int cols, const std::vector<int>& score_cols, const std::string& player) {
        bool in_score_col = false;
        for (int col : score_cols) if (x == col) { in_score_col = true; break; }
        if (!in_score_col) return true;
        
        int opponent_goal_row = (pushed_stone_owner == "circle") ? (rows - 3) : 2;
        return y != opponent_goal_row;
    }

    GameState applyMove(const GameState& state, const Move& move, const std::string& player) {
        GameState new_state = state;
        if (move.action == "move") {
            new_state.board[move.to[1]][move.to[0]] = state.board[move.from[1]][move.from[0]];
            new_state.board[move.from[1]][move.from[0]].clear();
        } else if (move.action == "push") {
            new_state.board[move.pushed_to[1]][move.pushed_to[0]] = state.board[move.to[1]][move.to[0]];
            new_state.board[move.to[1]][move.to[0]] = state.board[move.from[1]][move.from[0]];
            new_state.board[move.from[1]][move.from[0]].clear();
            if (new_state.board[move.to[1]][move.to[0]].count("side") && new_state.board[move.to[1]][move.to[0]].at("side") == "river") {
                new_state.board[move.to[1]][move.to[0]]["side"] = "stone";
                new_state.board[move.to[1]][move.to[0]].erase("orientation");
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
            new_state.board[fy][fx]["orientation"] = (state.board[fy][fx].at("orientation") == "horizontal") ? "vertical" : "horizontal";
        }
        return new_state;
    }
    
    void orderMovesForWinning(std::vector<Move>& moves, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        for (auto& move : moves) {
            double score = 0.0;
            if (isWinning(applyMove(state, move, player), player)) {
                move.evaluation_score = 100000.0;
                continue;
            }
            if (isDirectScoringMove(move, state, player)) score = 5000.0;

            if (move.action == "flip" && state.board[move.from[1]][move.from[0]].at("side") == "stone") {
                double flip_potential_score = 0.0;
                GameState next_state = applyMove(state, move, player);
                int flipped_x = move.from[0], flipped_y = move.from[1];

                for (auto [dx, dy] : std::vector<std::pair<int,int>>{{0,1}, {0,-1}, {1,0}, {-1,0}}) {
                    int adj_x = flipped_x - dx, adj_y = flipped_y - dy;
                    if (adj_x >= 0 && adj_x < state.cols && adj_y >= 0 && adj_y < state.rows &&
                        !state.board[adj_y][adj_x].empty() && state.board[adj_y][adj_x].at("owner") == player) {
                        auto destinations = traceAllRiverDestinations(next_state, flipped_x, flipped_y, adj_x, adj_y, player, player);
                        for (auto& dest : destinations) {
                            double dest_score = 0.0;
                            bool is_in_score_col = std::find(state.score_cols.begin(), state.score_cols.end(), dest.first) != state.score_cols.end();
                            if (dest.second == goal_row && is_in_score_col) dest_score = 600.0;
                            else if (std::abs(dest.second - goal_row) <= 1 && is_in_score_col) dest_score = 300.0;
                            else {
                                int progress = std::abs(adj_y - goal_row) - std::abs(dest.second - goal_row);
                                if (progress > 0) dest_score = 20.0 * progress;
                            }
                            if (dest_score > flip_potential_score) flip_potential_score = dest_score;
                        }
                    }
                }
                score += flip_potential_score;
            }
            move.evaluation_score = score + random_double(gen) * 0.1;
        }
        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.evaluation_score > b.evaluation_score;
        });
    }

    bool isDirectScoringMove(const Move& move, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        if (move.action == "move" && move.to[1] == goal_row) {
            for (int col : state.score_cols) {
                if (move.to[0] == col && !state.board[move.from[1]][move.from[0]].empty() &&
                    state.board[move.from[1]][move.from[0]].at("side") == "stone") {
                    return true;
                }
            }
        }
        return false;
    }
    
    void filterRecentMoves(std::vector<Move>& moves) {
        moves.erase(std::remove_if(moves.begin(), moves.end(), [this](const Move& m) { return isMoveRecent(m); }), moves.end());
    }

    bool isMoveRecent(const Move& move) {
        for (const auto& recent : recent_moves) {
            if (recent.action == move.action && recent.from == move.from && recent.to == move.to && recent.pushed_to == move.pushed_to) return true;
        }
        return false;
    }

    void addToHistory(const Move& move) {
        recent_moves.push_back(move);
        if (recent_moves.size() > HISTORY_SIZE) recent_moves.erase(recent_moves.begin());
    }

    double minimax(const GameState& state, int depth, double alpha, double beta,
                   bool maximizing_player, const std::chrono::steady_clock::time_point& start_time,
                   double time_limit) {

        if (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time).count() > time_limit) {
            return evaluateForWinning(state);
        }
        if (depth == 0 || isGameOver(state)) {
            return evaluateForWinning(state);
        }

        std::string current_player = maximizing_player ? side : opponent;
        auto moves = generateAllMoves(state, current_player);
        orderMovesForWinning(moves, state, current_player);
        if (moves.size() > 40 && depth > 1) moves.resize(40);

        if (maximizing_player) {
            double max_eval = -INFINITY_VAL;
            for (const auto& move : moves) {
                double eval = minimax(applyMove(state, move, current_player), depth - 1, alpha, beta, false, start_time, time_limit);
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break;
            }
            return max_eval;
        } else {
            double min_eval = INFINITY_VAL;
            for (const auto& move : moves) {
                double eval = minimax(applyMove(state, move, current_player), depth - 1, alpha, beta, true, start_time, time_limit);
                min_eval = std::min(min_eval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) break;
            }
            return min_eval;
        }
    }
    
    double evaluateForWinning(const GameState& state) {
        double score = 0.0;
        if (isWinning(state, side)) return INFINITY_VAL;
        if (isWinning(state, opponent)) return -INFINITY_VAL;

        score += WEIGHT_PIECES_IN_GOAL * (countPiecesInGoal(state, side) - countPiecesInGoal(state, opponent));
        score += WEIGHT_SCORING_THREATS * (countScoringThreats(state, side) - countScoringThreats(state, opponent));

        if (move_count > 4 && move_count < EARLY_GAME_MOVES) {
            score += WEIGHT_HIGHWAY_NETWORK * evaluateRiverNetwork(state, side);
            score += (WEIGHT_ADVANCEMENT * 0.5) * evaluateAdvancement(state);
        } else {
            score += WEIGHT_ADVANCEMENT * evaluateAdvancement(state);
        }

        score += WEIGHT_SPREAD * evaluatePieceSpread(state);
        score += WEIGHT_DEFENSE * evaluateDefense(state);
        score += WEIGHT_BOARD_CONTROL * evaluateBoardControl(state);
        return score;
    }

    // --- FIX: This function now prioritizes networks closer to the opponent's goal ---
    double evaluateRiverNetwork(const GameState& state, const std::string& player) {
        double total_network_score = 0;
        std::set<std::pair<int, int>> visited_rivers;
        int goal_row = (player == "circle") ? 2 : state.rows - 3;

        for (int y = 0; y < state.rows; ++y) {
            for (int x = 0; x < state.cols; ++x) {
                if (!state.board[y][x].empty() && state.board[y][x].at("owner") == player &&
                    state.board[y][x].at("side") == "river" && visited_rivers.find({x, y}) == visited_rivers.end()) {
                    
                    std::set<std::pair<int, int>> current_network_rivers, reachable_squares;
                    std::queue<std::pair<int, int>> q;
                    q.push({x, y});
                    visited_rivers.insert({x, y});
                    current_network_rivers.insert({x,y});

                    while (!q.empty()) {
                        auto [cx, cy] = q.front();
                        q.pop();
                        const auto& river_cell = state.board[cy][cx];
                        std::vector<std::pair<int, int>> dirs = (river_cell.at("orientation") == "horizontal") ? 
                                                                std::vector<std::pair<int, int>>{{-1, 0}, {1, 0}} : std::vector<std::pair<int, int>>{{0, -1}, {0, 1}};
                        for (auto [dx, dy] : dirs) {
                            int nx = cx + dx, ny = cy + dy;
                            while (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                                if (state.board[ny][nx].empty()) reachable_squares.insert({nx, ny});
                                else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("owner") == player &&
                                           state.board[ny][nx].at("side") == "river" && visited_rivers.find({nx, ny}) == visited_rivers.end()) {
                                    visited_rivers.insert({nx, ny});
                                    current_network_rivers.insert({nx,ny});
                                    q.push({nx, ny});
                                    break; 
                                } else break; 
                                nx += dx; ny += dy;
                            }
                        }
                    }

                    double network_advancement_bonus = 0;
                    for(const auto& river_pos : current_network_rivers) {
                        network_advancement_bonus += 15.0 / (std::abs(river_pos.second - goal_row) + 1);
                    }

                    total_network_score += network_advancement_bonus;
                    total_network_score += reachable_squares.size() * 0.5; // Reduce weight of raw size slightly
                }
            }
        }
        return total_network_score;
    }

    int countPiecesInGoal(const GameState& state, const std::string& player) {
        int count = 0;
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        for (int col : state.score_cols) {
            if (!state.board[goal_row][col].empty() && state.board[goal_row][col].at("owner") == player &&
                state.board[goal_row][col].at("side") == "stone") {
                count++;
            }
        }
        return count;
    }

    int countScoringThreats(const GameState& state, const std::string& player) {
        std::set<std::pair<int, int>> threatening_pieces;
        int goal_row = (player == "circle") ? 2 : state.rows - 3;

        std::vector<std::pair<int, int>> empty_goal_squares;
        for (int col : state.score_cols) {
            if (state.board[goal_row][col].empty()) {
                empty_goal_squares.push_back({col, goal_row});
            } else if (state.board[goal_row][col].at("owner") == player && state.board[goal_row][col].at("side") == "river") {
                threatening_pieces.insert({col, goal_row});
            }
        }
        if (empty_goal_squares.empty()) return threatening_pieces.size();

        for (const auto& goal_pos : empty_goal_squares) {
            int gx = goal_pos.first;
            int gy = goal_pos.second;

            std::set<std::pair<int, int>> visited_rivers_for_goal;
            std::queue<std::pair<int, int>> q;

            for (auto [dx, dy] : std::vector<std::pair<int,int>>{{0,1}, {0,-1}, {1,0}, {-1,0}}) {
                int nx = gx + dx, ny = gy + dy;
                if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows || state.board[ny][nx].empty()) continue;
                
                if (state.board[ny][nx].at("owner") == player) {
                    threatening_pieces.insert({nx, ny});
                }
                if (state.board[ny][nx].at("side") == "river" && visited_rivers_for_goal.find({nx, ny}) == visited_rivers_for_goal.end()) {
                    q.push({nx, ny});
                    visited_rivers_for_goal.insert({nx, ny});
                }
            }
            while(!q.empty()){
                auto [rx, ry] = q.front();
                q.pop();

                for (auto [dx, dy] : std::vector<std::pair<int,int>>{{0,1}, {0,-1}, {1,0}, {-1,0}}) {
                     int nx = rx + dx, ny = ry + dy;
                     if (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows &&
                         !state.board[ny][nx].empty() && state.board[ny][nx].at("owner") == player) {
                         threatening_pieces.insert({nx, ny});
                     }
                }
                const auto& river_cell = state.board[ry][rx];
                std::vector<std::pair<int, int>> flow_dirs = (river_cell.at("orientation") == "horizontal") ?
                                                               std::vector<std::pair<int, int>>{{-1, 0}, {1, 0}} : std::vector<std::pair<int, int>>{{0, -1}, {0, 1}};
                for(auto [dx, dy] : flow_dirs) {
                    int nx = rx + dx, ny = ry + dy;
                    while(nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                        const auto& next_cell = state.board[ny][nx];
                        if(!next_cell.empty() && next_cell.at("side") == "river") {
                            if(visited_rivers_for_goal.find({nx, ny}) == visited_rivers_for_goal.end()) {
                                q.push({nx, ny});
                                visited_rivers_for_goal.insert({nx, ny});
                            }
                            break;
                        } else if (!next_cell.empty()) break;
                        nx += dx; ny += dy;
                    }
                }
            }
        }
        return threatening_pieces.size();
    }

    // --- FIX: This function now evaluates the advancement of RIVERS as well as stones ---
    double evaluateAdvancement(const GameState& state) {
        double my_advancement = 0.0;
        int my_goal = (side == "circle") ? 2 : state.rows - 3;
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == side) { // Check applies to both stones and rivers
                    double weight = (cell.at("side") == "stone") ? 1.0 : 0.5; // Stones are slightly more important
                    my_advancement += weight * (10.0 / (std::abs(y - my_goal) + 1));
                }
            }
        }
        return my_advancement;
    }

    double evaluateDefense(const GameState& state) {
        double defense_score = 0.0;
        int our_goal_row = (side == "circle") ? 2 : state.rows - 3;
        for (int y = our_goal_row - 1; y <= our_goal_row + 1; y++) {
            if (y < 0 || y >= state.rows) continue;
            for (int x = 0; x < state.cols; x++) {
                if (!state.board[y][x].empty() && state.board[y][x].at("owner") == side) {
                    defense_score += 1.0;
                }
            }
        }
        return defense_score;
    }

    double evaluateBoardControl(const GameState& state) {
        double control_score = 0.0;
        int center_y = state.rows / 2, center_x = state.cols / 2;
        for (int y = center_y - 2; y <= center_y + 2; y++) {
            if (y < 0 || y >= state.rows) continue;
            for (int x = center_x - 2; x <= center_x + 2; x++) {
                if (x < 0 || x >= state.cols) continue;
                if (!state.board[y][x].empty() && state.board[y][x].at("owner") == side) {
                   control_score += 1.0;
                }
            }
        }
        return control_score;
    }

    double evaluatePieceSpread(const GameState& state) {
        std::set<int> my_columns;
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                if (!state.board[y][x].empty() && state.board[y][x].at("owner") == side) {
                    my_columns.insert(x);
                }
            }
        }
        return static_cast<double>(my_columns.size());
    }

    bool isWinning(const GameState& state, const std::string& player) {
        return countPiecesInGoal(state, player) >= 4;
    }

    bool isGameOver(const GameState& state) {
        return isWinning(state, side) || isWinning(state, opponent);
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