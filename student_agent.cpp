#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <queue>
#include <set>
#include <unordered_map>
#include <cmath>
#include <iostream>

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

// River Network Structure for Strategic Planning
struct RiverNetwork {
    std::vector<std::pair<int,int>> river_positions;
    std::vector<std::pair<int,int>> endpoints;
    double score_area_distance;
    int network_size;
    bool connects_to_goal;
};

// Game phase enumeration
enum GamePhase {
    EARLY_GAME,  // Focus on network building
    MID_GAME,    // Balanced approach
    LATE_GAME    // Focus on scoring
};

class StudentAgent {
private:
    std::string side;
    std::string opponent;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> random_double;
    
    int move_count = 0;
    std::pair<int,int> stab1, stab2;
    
    // Move history to prevent repetition
    std::vector<Move> recent_moves;
    static constexpr int HISTORY_SIZE = 4;
    
    // Performance constants
    static constexpr int MAX_DEPTH = 3;
    static constexpr double INFINITY_VAL = 1000000.0;
    static constexpr double TIME_BUFFER = 0.3;
    static constexpr double MAX_MOVE_TIME = 0.6;
    static constexpr int MAX_MOVES_PER_POSITION = 80;
    
    // Base evaluation weights (will be adjusted dynamically)
    static constexpr double WEIGHT_PIECES_IN_GOAL = 1000.0;
    static constexpr double WEIGHT_PIECES_NEAR_GOAL = 120.0;
    static constexpr double WEIGHT_ADVANCEMENT = 25.0;
    static constexpr double WEIGHT_SPREAD = 15.0;
    static constexpr double WEIGHT_RIVER_PUSH = 85.0;
    static constexpr double WEIGHT_DEFENSE = 20.0;
    static constexpr double WEIGHT_BOARD_CONTROL = 10.0;
    
    // Targeted River Safety Weight
    static constexpr double WEIGHT_DANGER_ZONE_PENALTY = -100.0;

public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()), random_double(0.0, 1.0) {
        opponent = (this->side == "circle") ? "square" : "circle";
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>>& board,
                int rows, int cols, const std::vector<int>& score_cols,
                float current_player_time, float opponent_time) {
        
        auto start_time = std::chrono::steady_clock::now();
        GameState state(board, side, rows, cols, score_cols);
        
        if(move_count < 4){
            if(state.current_player == "circle"){
                int row = state.rows-3;
                int col1 = score_cols[0]-1;
                int col2 = score_cols.back()+1;
                stab1 = {col1,row};
                stab2 = {col2,row};
                if(move_count==0){
                    move_count++;
                    return {"move", {col1, row-1}, {col1, row}, {}, ""};
                } else if(move_count==1){
                    move_count++;
                    return {"move", {col2, row-1}, {col2, row}, {}, ""};
                } else if(move_count==2){
                    move_count++;
                    return {"flip", {col1, row}, {col1, row}, {}, "vertical"};
                } else if(move_count==3){
                    move_count++;
                    return {"flip", {col2, row}, {col2, row}, {}, "vertical"};
                }
            }
            else{
                int row = 2;
                int col1 = score_cols[0]-1;
                int col2 = score_cols.back()+1;
                stab1 = {col1, row};
                stab2 = {col2, row};
                if(move_count==0){
                    move_count++;
                    return {"move", {col1, row+1}, {col1, row}, {}, ""};
                } else if(move_count==1){
                    move_count++;
                    return {"move", {col2, row+1}, {col2, row}, {}, ""};
                } else if(move_count==2){
                    move_count++;
                    return {"flip", {col1, row}, {col1, row}, {}, "vertical"};
                } else if(move_count==3){
                    move_count++;
                    return {"flip", {col2, row}, {col2, row}, {}, "vertical"};
                }
            }
        }
        
        // Check for immediate wins first
        auto moves = generateAllMoves(state, side);
        for (const auto& move : moves) {
            GameState next_state = applyMove(state, move, side);
            if (isWinning(next_state, side)) {
                return move;
            }
        }
        
        if (moves.empty()) {
            return {"move", {0,0}, {0,0}, {}, ""};
        }
        
        // Time budget
        double move_budget = std::min(MAX_MOVE_TIME, std::max(0.05, current_player_time * 0.015));
        Move best_move;
        double best_score = -INFINITY_VAL;
        
        // Remove recent moves to prevent loops
        filterRecentMoves(moves);
        
        // Strategic move ordering with dynamic weights
        orderMovesForWinning(moves, state, side);
        
        // Iterative deepening
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
                move_score += (random_double(gen) - 0.5) * 0.2;
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
    // NEW: Game phase detection for dynamic weight adjustment
    GamePhase getGamePhase(const GameState& state) {
        // Count total pieces and network size
        int our_pieces = 0;
        int opponent_pieces = 0;
        int our_network_size = 0;
        int our_pieces_near_goal = 0;
        int opponent_pieces_near_goal = 0;
        
        int our_goal_row = (side == "circle") ? 2 : state.rows - 3;
        int opponent_goal_row = (opponent == "circle") ? 2 : state.rows - 3;
        
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty()) continue;
                
                if (cell.at("owner") == side) {
                    our_pieces++;
                    if (cell.at("side") == "river") {
                        our_network_size++;
                    }
                    if (std::abs(y - our_goal_row) <= 3) {
                        our_pieces_near_goal++;
                    }
                } else if (cell.at("owner") == opponent) {
                    opponent_pieces++;
                    if (std::abs(y - opponent_goal_row) <= 3) {
                        opponent_pieces_near_goal++;
                    }
                }
            }
        }
        
        // Phase detection logic
        if (our_network_size < 8 || our_pieces < 10) {
            return EARLY_GAME;  // Focus on building network
        } else if (our_pieces_near_goal < 6 || opponent_pieces_near_goal >= 4) {
            return MID_GAME;    // Balanced approach, react to threats
        } else {
            return LATE_GAME;   // Focus on scoring
        }
    }
    
    // NEW: Get dynamic weights based on game phase
    void getDynamicWeights(const GameState& state, double& river_chaining, double& strategic_flip, 
                          double& river_transport, double& network_extension, double& river_corridor) {
        GamePhase phase = getGamePhase(state);
        
        switch (phase) {
            case EARLY_GAME:
                // High focus on network building
                river_chaining = 120.0;
                strategic_flip = 250.0;
                river_transport = 300.0;
                network_extension = 300.0;
                river_corridor = 180.0;
                break;
                
            case MID_GAME:
                // Balanced approach with transport priority
                river_chaining = 100.0;
                strategic_flip = 150.0;
                river_transport = 220.0;  // Higher transport priority
                network_extension = 180.0;
                river_corridor = 140.0;
                break;
                
            case LATE_GAME:
                // Focus on scoring, minimal network building
                river_chaining = 80.0;
                strategic_flip = 80.0;
                river_transport = 280.0;  // Highest transport priority
                network_extension = 100.0;
                river_corridor = 80.0;
                break;
        }
    }

    // Check if vertical river is in specific danger zone only
    bool isVerticalRiverInDangerZone(const GameState& state, int x, int y, const std::string& orientation) {
        if (orientation != "vertical") return false;
        
        int opponent_goal_row = (opponent == "circle") ? 2 : state.rows - 3;
        int distance_to_opponent_goal = std::abs(y - opponent_goal_row);
        
        if (distance_to_opponent_goal <= 1) {
            for (int col : state.score_cols) {
                if (x == col) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // Check if river landing position creates strategic network connectivity
    bool doesRiverCreateNetworkConnectivity(const GameState& state, int x, int y, const std::string& player) {
        int adjacent_rivers = 0;
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto [dx, dy] : dirs) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                const auto& cell = state.board[ny][nx];
                if (!cell.empty() && cell.at("owner") == player && cell.at("side") == "river") {
                    adjacent_rivers++;
                }
            }
        }
        
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        bool near_goal = std::abs(y - goal_row) <= 3;
        
        return (adjacent_rivers >= 2) || (adjacent_rivers >= 1 && near_goal);
    }

    // Analyze river networks for strategic planning
    std::vector<RiverNetwork> analyzeRiverNetworks(const GameState& state, const std::string& player) {
        std::vector<RiverNetwork> networks;
        std::set<std::pair<int,int>> visited;
        
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                if (visited.count({x,y})) continue;
                
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == player && cell.at("side") == "river") {
                    RiverNetwork network = traceConnectedRivers(state, x, y, visited, player);
                    if (network.network_size > 0) {
                        networks.push_back(network);
                    }
                }
            }
        }
        return networks;
    }
    
    // Trace a connected river network starting from a position
    RiverNetwork traceConnectedRivers(const GameState& state, int start_x, int start_y, 
                                    std::set<std::pair<int,int>>& visited, const std::string& player) {
        RiverNetwork network;
        std::queue<std::pair<int,int>> queue;
        queue.push({start_x, start_y});
        
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        double min_goal_distance = 1000.0;
        
        while (!queue.empty()) {
            auto [x, y] = queue.front();
            queue.pop();
            
            if (visited.count({x,y})) continue;
            visited.insert({x,y});
            
            const auto& cell = state.board[y][x];
            if (cell.empty() || cell.at("owner") != player || cell.at("side") != "river") continue;
            
            network.river_positions.push_back({x, y});
            
            double goal_dist = std::abs(y - goal_row);
            for (int col : state.score_cols) {
                if (x == col) {
                    goal_dist = std::min(goal_dist, (double)std::abs(y - goal_row));
                    if (y == goal_row) network.connects_to_goal = true;
                }
            }
            min_goal_distance = std::min(min_goal_distance, goal_dist);
            
            // Explore connected rivers via flow
            auto destinations = traceAllRiverDestinations(state, x, y, x, y, player, player);
            for (auto [dest_x, dest_y] : destinations) {
                const auto& dest_cell = state.board[dest_y][dest_x];
                if (!dest_cell.empty() && dest_cell.at("owner") == player && 
                    dest_cell.at("side") == "river" && !visited.count({dest_x, dest_y})) {
                    queue.push({dest_x, dest_y});
                }
            }
            
            std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
            for (auto [dx, dy] : dirs) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows && !visited.count({nx, ny})) {
                    const auto& adj_cell = state.board[ny][nx];
                    if (!adj_cell.empty() && adj_cell.at("owner") == player && adj_cell.at("side") == "river") {
                        queue.push({nx, ny});
                    }
                }
            }
        }
        
        network.network_size = network.river_positions.size();
        network.score_area_distance = min_goal_distance;
        return network;
    }
    
    // Check if a position would strategically extend a river network
    bool isStrategicRiverPosition(const GameState& state, int x, int y, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        bool moves_toward_goal = std::abs(y - goal_row) <= 3;
        
        int adjacent_rivers = 0;
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto [dx, dy] : dirs) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                const auto& cell = state.board[ny][nx];
                if (!cell.empty() && cell.at("owner") == player && cell.at("side") == "river") {
                    adjacent_rivers++;
                }
            }
        }
        
        return (adjacent_rivers >= 1 && moves_toward_goal) || adjacent_rivers >= 2;
    }
    
    // River chaining evaluation with dynamic weights
    double evaluateRiverChaining(const GameState& state) {
        double score = 0.0;
        
        // Get dynamic weights based on game phase
        double river_chaining, strategic_flip, river_transport, network_extension, river_corridor;
        getDynamicWeights(state, river_chaining, strategic_flip, river_transport, network_extension, river_corridor);
        
        auto our_networks = analyzeRiverNetworks(state, side);
        auto opponent_networks = analyzeRiverNetworks(state, opponent);
        
        for (const auto& network : our_networks) {
            score += river_chaining * network.network_size;
            
            if (network.connects_to_goal) {
                score += river_corridor;
            }
            
            score += network_extension / (network.score_area_distance + 1.0);
            
            if (network.network_size >= 4) {
                score += river_corridor * 0.5;
            }
        }
        
        for (const auto& network : opponent_networks) {
            score -= river_chaining * 0.6 * network.network_size;
            if (network.connects_to_goal) {
                score -= river_corridor * 0.8;
            }
        }
        
        // Only penalize rivers in specific danger zones
        for (int y = 0; y < state.rows; y++) {
            for (int x = 0; x < state.cols; x++) {
                const auto& cell = state.board[y][x];
                if (!cell.empty() && cell.at("owner") == side && cell.at("side") == "river") {
                    if (isVerticalRiverInDangerZone(state, x, y, cell.at("orientation"))) {
                        score += WEIGHT_DANGER_ZONE_PENALTY;
                    }
                }
            }
        }
        
        return score;
    }

    // Generate all moves with strategic river logic
    std::vector<Move> generateAllMoves(const GameState& state, const std::string& player) {
        std::vector<Move> moves;
        for (int y = 0; y < state.rows && moves.size() < MAX_MOVES_PER_POSITION * 4; y++) {
            for (int x = 0; x < state.cols && moves.size() < MAX_MOVES_PER_POSITION * 4; x++) {
                const auto& cell = state.board[y][x];
                if (cell.empty() || cell.at("owner") != player) continue;
                
                std::string side_type = cell.at("side");
                if (side_type == "stone") {
                    generateStoneMoves(moves, state, x, y, player);
                    if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"flip", {x,y}, {x,y}, {}, "horizontal"});
                    if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"flip", {x,y}, {x,y}, {}, "vertical"});
                } else if (side_type == "river") {
                    generateRiverMoves(moves, state, x, y, player);
                    generateRiverPushMoves(moves, state, x, y, player);
                    if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"rotate", {x,y}, {x,y}, {}, ""});
                    if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"flip", {x,y}, {x,y}, {}, ""});
                }
                
                if (moves.size() >= MAX_MOVES_PER_POSITION * 4) break;
            }
        }
        return moves;
    }
    
    // FIXED: Enhanced river flow exploration that works for ALL pieces (including rivers)
    std::vector<std::pair<int,int>> traceAllRiverDestinations(const GameState& state,
                                                             int start_x, int start_y, int ox, int oy,
                                                             const std::string& moving_piece_owner, const std::string& player,
                                                             bool is_river_push = false) {
        std::vector<std::pair<int,int>> destinations;
        std::set<std::pair<int,int>> visited;
        std::queue<std::pair<int,int>> queue;
        queue.push({start_x, start_y});
        
        while (!queue.empty()) {
            auto [x, y] = queue.front();
            queue.pop();
            
            if (visited.count({x, y}) || x < 0 || x >= state.cols || y < 0 || y >= state.rows) {
                continue;
            }
            
            visited.insert({x, y});
            const auto& cell = state.board[y][x];
            
            // Handle empty cells - pieces can land here
            if (cell.empty()) {
                // For rivers moving through network, they can land anywhere empty
                // For stones, check push restrictions
                if (moving_piece_owner == player || 
                    canPushStoneTo(x, y, moving_piece_owner, state.rows, state.cols, state.score_cols, player)) {
                    destinations.push_back({x, y});
                }
                continue;
            }
            
            // Handle stones - they block flow (except at starting position)
            if (!cell.empty() && cell.at("side") == "stone") {
                if(is_river_push && x==start_x && y==start_y) {
                    const auto& river_cell = state.board[oy][ox];
                    std::string orientation = river_cell.at("orientation");
                    std::vector<std::pair<int,int>> flow_dirs;
                    if (orientation == "horizontal") {
                        flow_dirs = {{1,0}, {-1,0}};
                    } else {
                        flow_dirs = {{0,1}, {0,-1}};
                    }
                    
                    for (auto [dx, dy] : flow_dirs) {
                        int nx = x + dx;
                        int ny = y + dy;
                        while (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                            if (visited.count({nx, ny})) break;
                            const auto& next_cell = state.board[ny][nx];
                            if (next_cell.empty()) {
                                if (moving_piece_owner == player || 
                                    canPushStoneTo(nx, ny, moving_piece_owner, state.rows, state.cols, state.score_cols, player)) {
                                    destinations.push_back({nx, ny});
                                }
                                nx += dx;
                                ny += dy;
                                continue;
                            } else if (nx==ox && ny==oy) {
                                nx+=dx;
                                ny+=dy;
                                continue;
                            } else if (!next_cell.empty() && next_cell.at("side") == "stone") {
                                break;
                            } else if (!next_cell.empty() && next_cell.at("side") == "river") {
                                if (!visited.count({nx, ny})) {
                                    queue.push({nx, ny});
                                }
                                break;
                            } else {
                                break;
                            }
                        }
                    }
                }
                continue;
            }
            
            // Handle rivers - they redirect flow and allow continued transport
            if (!cell.empty() && cell.at("side") == "river") {
                std::string orientation = cell.at("orientation");
                std::vector<std::pair<int,int>> flow_dirs;
                if (orientation == "horizontal") {
                    flow_dirs = {{1,0}, {-1,0}};
                } else {
                    flow_dirs = {{0,1}, {0,-1}};
                }
                
                // Continue flow in river direction
                for (auto [dx, dy] : flow_dirs) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    while (nx >= 0 && nx < state.cols && ny >= 0 && ny < state.rows) {
                        if (visited.count({nx, ny})) break;
                        const auto& next_cell = state.board[ny][nx];
                        
                        if (next_cell.empty()) {
                            // FIXED: Rivers can travel to empty spaces
                            if (moving_piece_owner == player || 
                                canPushStoneTo(nx, ny, moving_piece_owner, state.rows, state.cols, state.score_cols, player)) {
                                destinations.push_back({nx, ny});
                            }
                            nx += dx;
                            ny += dy;
                            continue;
                        } else if (nx==ox && ny==oy) {
                            nx+=dx;
                            ny+=dy;
                            continue;
                        } else if (!next_cell.empty() && next_cell.at("side") == "stone") {
                            break;
                        } else if (!next_cell.empty() && next_cell.at("side") == "river") {
                            // Continue through connected rivers
                            if (!visited.count({nx, ny})) {
                                queue.push({nx, ny});
                            }
                            break;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        return destinations;
    }
    
    // Stone moves with multiple river destinations
    void generateStoneMoves(std::vector<Move>& moves, const GameState& state,
                           int x, int y, const std::string& player) {
        std::vector<std::pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto [dx, dy] : dirs) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;
            
            if (state.board[ny][nx].empty()) {
                if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"move", {x,y}, {nx,ny}, {}, ""});
            } else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("side") == "river") {
                // Stone can use river network
                auto destinations = traceAllRiverDestinations(state, nx, ny, x, y, player, player);
                for (auto [dest_x, dest_y] : destinations) {
                    if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"move", {x,y}, {dest_x,dest_y}, {}, ""});
                }
            } else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("side") == "stone" ) {
                int nx2 = x + 2*dx, ny2 = y + 2*dy;
                std::string pushed_stone_owner = state.board[ny][nx].at("owner");
                if (nx2 >= 0 && ny2 >= 0 && nx2 < state.cols && ny2 < state.rows &&
                    state.board[ny2][nx2].empty() &&
                    canPushStoneTo(nx2, ny2, pushed_stone_owner, state.rows, state.cols, state.score_cols, player)) {
                    if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"push", {x,y}, {nx,ny}, {nx2,ny2}, ""});
                }
            }
        }
    }
    
    // ENHANCED: River moves with long-distance network transport
    void generateRiverMoves(std::vector<Move>& moves, const GameState& state,
                           int x, int y, const std::string& player) {
        std::vector<std::pair<int,int>> all_dirs = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto [dx, dy] : all_dirs) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;
            
            if (state.board[ny][nx].empty()) {
                // Adjacent river move
                if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"move", {x,y}, {nx,ny}, {}, ""});
            } else if (!state.board[ny][nx].empty() && state.board[ny][nx].at("side") == "river") {
                // FIXED: Rivers can use river network to transport themselves!
                auto destinations = traceAllRiverDestinations(state, nx, ny, x, y, player, player);
                for (auto [dest_x, dest_y] : destinations) {
                    // Only add long-distance moves (distance > 2) to avoid duplicating adjacent moves
                    int distance = std::abs(dest_x - x) + std::abs(dest_y - y);
                    if (distance > 2) {
                        if(std::make_pair(x,y)!=stab1 && std::make_pair(x,y)!=stab2) moves.push_back({"move", {x,y}, {dest_x,dest_y}, {}, ""});
                    }
                }
            }
        }
    }
    
    // River push with multiple destination choices
    void generateRiverPushMoves(std::vector<Move>& moves, const GameState& state,
                               int river_x, int river_y, const std::string& player) {
        const auto& river_cell = state.board[river_y][river_x];
        
        std::vector<std::pair<int,int>> adjacent_dirs = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto [dx, dy] : adjacent_dirs) {
            int adj_x = river_x + dx;
            int adj_y = river_y + dy;
            if (adj_x < 0 || adj_x >= state.cols || adj_y < 0 || adj_y >= state.rows) continue;
            if (state.board[adj_y][adj_x].empty()) continue;
            
            const auto& adjacent_piece = state.board[adj_y][adj_x];
            if (adjacent_piece.at("side") != "stone") continue;
            
            std::string pushed_stone_owner = adjacent_piece.at("owner");
            
            auto destinations = traceAllRiverDestinations(state, adj_x, adj_y, river_x, river_y, pushed_stone_owner, player, true);
            
            for (auto [dest_x, dest_y] : destinations) {
                if (dest_x != river_x || dest_y != river_y) {
                    if(std::make_pair(river_x,river_y)!=stab1 && std::make_pair(river_x,river_y)!=stab2) moves.push_back({
                        "push",
                        {river_x, river_y},
                        {adj_x, adj_y},
                        {dest_x, dest_y},
                        river_cell.at("orientation")
                    });
                }
            }
        }
    }
    
    // Push restriction based on pushed stone's owner
    bool canPushStoneTo(int x, int y, const std::string& pushed_stone_owner,
                       int rows, int cols, const std::vector<int>& score_cols, const std::string& player) {
        bool in_score_col = false;
        for (int col : score_cols) {
            if (x == col) {
                in_score_col = true;
                break;
            }
        }
        
        if (!in_score_col) return true;
        
        std::string opponent = (player == "circle") ? "square" : "circle";
        
        if (pushed_stone_owner == opponent) {
            return (y != (rows - 3) && y != (rows - 2));
        } else {
            if(pushed_stone_owner == "circle")
                return (y != (rows - 3));
            else
                return (y != 2);
        }
    }
    
    // Apply move
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
            
            if (!new_state.board[ty][tx].empty() && new_state.board[ty][tx].at("side") == "river") {
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
    
    // ENHANCED: Strategic move ordering with dynamic weights based on game phase
    void orderMovesForWinning(std::vector<Move>& moves, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        
        // Get dynamic weights based on game phase
        double river_chaining, strategic_flip, river_transport, network_extension, river_corridor;
        getDynamicWeights(state, river_chaining, strategic_flip, river_transport, network_extension, river_corridor);
        
        for (auto& move : moves) {
            double score = 0.0;
            
            if (isImmediateWinningMove(move, state, player)) {
                score = 10000.0;
                move.evaluation_score = score;
                continue;
            }
            
            if (isDirectScoringMove(move, state, player)) {
                score = 5000.0;
            }
            
            // Dynamic river flip evaluation based on game phase
            if (move.action == "flip") {
                int flip_x = move.from[0], flip_y = move.from[1];
                const auto& piece = state.board[flip_y][flip_x];
                
                if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "stone") {
                    if (isStrategicRiverPosition(state, flip_x, flip_y, player)) {
                        score += strategic_flip;  // Dynamic weight based on game phase
                        
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += strategic_flip * 0.5;
                                break;
                            }
                        }
                        
                        if (std::abs(flip_y - goal_row) <= 2) {
                            score += strategic_flip * 0.8;
                        }
                    }
                    
                    if (isVerticalRiverInDangerZone(state, flip_x, flip_y, move.orientation)) {
                        score += WEIGHT_DANGER_ZONE_PENALTY;
                    }
                } else if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "river") {
                    if (flip_y == goal_row) {
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += 8000.0;
                                break;
                            }
                        }
                    }
                    int distance_to_goal = std::abs(flip_y - goal_row);
                    if (distance_to_goal <= 1) {
                        score += 2000.0;
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += 1000.0;
                                break;
                            }
                        }
                    } else if (distance_to_goal <= 2) {
                        score += 800.0;
                        for (int col : state.score_cols) {
                            if (flip_x == col) {
                                score += 400.0;
                                break;
                            }
                        }
                    }
                } else {
                    score += 15.0;
                    int distance_to_goal = std::abs(flip_y - goal_row);
                    if (distance_to_goal <= 3) {
                        score -= 50.0;
                    }
                }
            } 
            // ENHANCED: Move evaluation with prioritized river transport
            else if (move.action == "move") {
                int from_x = move.from[0], from_y = move.from[1];
                int to_x = move.to[0], to_y = move.to[1];
                
                // Check if this is a river being transported
                const auto& moving_piece = state.board[from_y][from_x];
                if (!moving_piece.empty() && moving_piece.at("owner") == player && 
                    moving_piece.at("side") == "river") {
                    
                    // PRIORITIZE: River transportation through network
                    int transport_distance = std::abs(to_x - from_x) + std::abs(to_y - from_y);
                    if (transport_distance >= 3) { // Long-distance river transport via network
                        score += river_transport;  // Dynamic weight - highest in late game
                        
                        // MAJOR BONUS: Rivers transported far distances
                        if (transport_distance >= 5) {
                            score += river_transport * 0.5;  // Extra bonus for very long transport
                        }
                        
                        // EXTRA BONUS: River landing at strategic network extension point
                        if (isStrategicRiverPosition(state, to_x, to_y, player)) {
                            score += network_extension;
                        }
                        
                        // BONUS: River creating network connectivity
                        if (doesRiverCreateNetworkConnectivity(state, to_x, to_y, player)) {
                            score += river_transport * 0.4;
                        }
                        
                        // BONUS: River moving toward goal area
                        int old_distance = std::abs(from_y - goal_row);
                        int new_distance = std::abs(to_y - goal_row);
                        if (new_distance < old_distance) {
                            score += river_transport * 0.6;
                        }
                        
                        // EXTRA BONUS: Rivers in score columns
                        for (int col : state.score_cols) {
                            if (to_x == col) {
                                score += river_transport * 0.3;
                                break;
                            }
                        }
                    } else {
                        // Adjacent river moves get moderate bonus
                        score += 100.0;
                    }
                } else {
                    // Stone move evaluation (lower priority than river transport)
                    int target_y = move.to[1];
                    int source_y = move.from[1];
                    int target_x = move.to[0];
                    
                    if (target_y == goal_row) {
                        for (int col : state.score_cols) {
                            if (target_x == col) {
                                const auto& piece = state.board[source_y][move.from[0]];
                                if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "stone") {
                                    score += 10000.0;
                                } else {
                                    score += 7000.0;
                                }
                                break;
                            }
                        }
                    }
                    
                    int old_distance = std::abs(source_y - goal_row);
                    int new_distance = std::abs(target_y - goal_row);
                    int progress = old_distance - new_distance;
                    
                    if (progress > 0) {
                        score += 150.0 * progress;
                        if (new_distance == 0) {
                            score += 1000.0;
                        } else if (new_distance == 1) {
                            score += 600.0;
                        } else if (new_distance <= 2) {
                            score += 300.0;
                        }
                    }
                    
                    for (int col : state.score_cols) {
                        if (target_x == col) {
                            score += 100.0;
                            break;
                        }
                    }
                    
                    int move_distance = std::abs(target_x - move.from[0]) + std::abs(target_y - move.from[1]);
                    if (move_distance > 2) {
                        score += 40.0; // Stone transport bonus (much lower than river transport)
                    }
                    
                    if (isStrategicRiverPosition(state, target_x, target_y, player)) {
                        score += 400.0;
                    }
                }
            } else if (move.action == "push") {
                score += 50.0;
                
                int pushed_x = move.pushed_to[0], pushed_y = move.pushed_to[1];
                
                if (isStrategicRiverPosition(state, pushed_x, pushed_y, player)) {
                    score += 600.0;
                }
                
                int from_x = move.from[0], from_y = move.from[1];
                if (!state.board[from_y][from_x].empty() && state.board[from_y][from_x].at("side") == "river") {
                    score += 120.0;
                    
                    int push_distance = std::abs(move.pushed_to[0] - move.to[0]) + std::abs(move.pushed_to[1] - move.to[1]);
                    if (push_distance > 2) {
                        score += 50.0;
                    }
                    
                    if (push_distance >= 3) {
                        score += river_chaining * 0.7;
                    }
                    
                    evaluateRiverPush(score, move, state, player);
                } else {
                    int target_y = move.to[1];
                    int pushed_y = move.pushed_to[1];
                    if (std::abs(pushed_y - goal_row) < std::abs(target_y - goal_row)) {
                        score += 200.0;
                    }
                }
            } else if (move.action == "rotate") {
                score += 10.0;
            }
            
            score += random_double(gen) * 2.0;
            move.evaluation_score = score;
        }
        
        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.evaluation_score > b.evaluation_score;
        });
    }
    
    // Helper functions (unchanged from original)
    bool isImmediateWinningMove(const Move& move, const GameState& state, const std::string& player) {
        GameState next_state = applyMove(state, move, player);
        return isWinning(next_state, player);
    }
    
    bool isDirectScoringMove(const Move& move, const GameState& state, const std::string& player) {
        int goal_row = (player == "circle") ? 2 : state.rows - 3;
        
        if (move.action == "move" && move.to[1] == goal_row) {
            for (int col : state.score_cols) {
                if (move.to[0] == col) {
                    const auto& piece = state.board[move.from[1]][move.from[0]];
                    if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "stone") {
                        return true;
                    }
                }
            }
        }
        if (move.action == "flip" && move.from[1] == goal_row) {
            for (int col : state.score_cols) {
                if (move.from[0] == col) {
                    const auto& piece = state.board[move.from[1]][move.from[0]];
                    if (!piece.empty() && piece.at("owner") == player && piece.at("side") == "river") {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
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
                int old_dist = std::abs(piece_y - opponent_goal);
                int new_dist = std::abs(final_y - opponent_goal);
                if (new_dist < old_dist) score += 40.0;
                
                for (int col : state.score_cols) {
                    if (final_x == col) {
                        score += 30.0;
                        break;
                    }
                }
            } else {
                int old_dist = std::abs(piece_y - goal_row);
                int new_dist = std::abs(final_y - goal_row);
                if (new_dist > old_dist) score += 50.0;
            }
        }
    }
    
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
        
        if (moves.size() > 40 && depth > 1) {
            orderMovesForWinning(moves, state, current_player);
            moves.resize(40);
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
    
    // Evaluation function with dynamic river safety
    double evaluateForWinning(const GameState& state) {
        double score = 0.0;
        
        int my_pieces_in_goal = countPiecesInGoal(state, side);
        int opponent_pieces_in_goal = countPiecesInGoal(state, opponent);
        
        if (my_pieces_in_goal >= 4) return INFINITY_VAL;
        if (opponent_pieces_in_goal >= 4) return -INFINITY_VAL;
        
        score += WEIGHT_PIECES_IN_GOAL * (my_pieces_in_goal - opponent_pieces_in_goal);
        
        int my_pieces_near_goal = countPiecesNearGoal(state, side);
        int opponent_pieces_near_goal = countPiecesNearGoal(state, opponent);
        score += WEIGHT_PIECES_NEAR_GOAL * (my_pieces_near_goal - opponent_pieces_near_goal);
        
        score += WEIGHT_ADVANCEMENT * evaluateAdvancement(state);
        score += WEIGHT_SPREAD * evaluatePieceSpread(state);
        score += WEIGHT_DEFENSE * evaluateDefense(state);
        score += WEIGHT_BOARD_CONTROL * evaluateBoardControl(state);
        
        // River evaluation with dynamic weights
        score += evaluateRiverChaining(state);
        
        return score;
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
        
        int col = state.score_cols.front() - 1;
        if (!state.board[goal_row][col].empty() &&
            state.board[goal_row][col].at("owner") == player &&
            state.board[goal_row][col].at("side") == "stone") {
            count++;
        }
        
        col = state.score_cols.back() + 1;
        if (!state.board[goal_row][col].empty() &&
            state.board[goal_row][col].at("owner") == player &&
            state.board[goal_row][col].at("side") == "stone") {
            count++;
        }
        
        for (int col : state.score_cols) {
            if (!state.board[goal_row][col].empty() &&
                state.board[goal_row][col].at("owner") == player &&
                state.board[goal_row][col].at("side") == "river") {
                count++;
            }
        }
        return count;
    }
    
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
                    my_advancement += 8.0 / (distance_to_goal + 1);
                    
                    for (int col : state.score_cols) {
                        if (x == col) {
                            my_advancement += 15.0 / (distance_to_goal + 1);
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