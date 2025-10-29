from mpi4py import profile
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from typing import Dict, List, Tuple, Optional

class DartsELOCalculator:
    def __init__(self, initial_elo: int = 1500, k_factor: int = 24, k_factor_major: int = 40):
        """
        Initialize ELO calculator for darts players.
        
        Args:
            initial_elo: Starting ELO for new players
            k_factor: Standard K-factor for rating updates
            k_factor_major: Higher K-factor for major tournaments
        """
        self.initial_elo = initial_elo
        self.k_factor = k_factor # Standard K-factor (deepSeek suggested 32)
        self.k_factor_major = k_factor_major # K-factor for major tournaments (deepSeek suggested 48)
        self.player_elos = {}  # Current ELO ratings
        self.elo_history = []  # Historical ELO data
        self._player_games = {}  # Track games efficiently
    
    # def get_k_factor(self, event_name: str, is_final: bool = False) -> int:
    #     """Determine K-factor based on tournament importance and stage."""
    #     k = self.k_factor_major if self.is_major_tournament(event_name) else self.k_factor
    #     # Increase K-factor for finals
    #     if is_final and 'final' in event_name.lower():
    #         k = int(k * 1.5)
    #     return k

    def get_k_factor(self, player_name: str = "") -> int:
        """Determine K-factor for a player.

        Uses number of previously processed matches (self.elo_history) and current ELO.
        New/low-rated players get the higher k_factor_major.
        """
        # safe current elo (default to initial_elo for unknown players)
        player_elo = self.player_elos.get(player_name, self.initial_elo)

        # count previously processed matches for this player
        games_played = sum(
            1 for h in self.elo_history
            if h.get('player1') == player_name or h.get('player2') == player_name
        )

        # Higher K for inexperienced players or low-rated players
        if games_played < 30 and player_elo < 2300:
            return self.k_factor_major
        return self.k_factor

    def get_player_elo(self, player_name: str) -> int:
        """Get current ELO for a player, initialize if new."""
        if player_name not in self.player_elos:
            self.player_elos[player_name] = self.initial_elo
        return self.player_elos[player_name]
    
    def calculate_expected_score(self, elo_a: int, elo_b: int) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    # def update_elo(self, winner: str, loser: str, event_name: str, 
    #                round_name: str, date: str, score: Tuple[int, int]) -> None:
    #     """Update ELO ratings after a match."""

    #     # Get current ELOs
    #     winner_elo = self.get_player_elo(winner)
    #     loser_elo = self.get_player_elo(loser)
        
    #     # # Determine if this is a final
    #     # is_final = 'final' in round_name.lower() if round_name else False
        
    #     # # Get appropriate K-factor
    #     # k = self.get_k_factor()
    #     k_winner = self.get_k_factor(winner)
    #     k_loser = self.get_k_factor(loser)
        
    #     # Calculate expected scores
    #     expected_winner = self.calculate_expected_score(winner_elo, loser_elo)
    #     expected_loser = self.calculate_expected_score(loser_elo, winner_elo)
        
    #     # Update ELOs (winner gets 1 point, loser gets 0)
    #     winner_new_elo = winner_elo + k_winner * (1 - expected_winner)
    #     loser_new_elo = loser_elo + k_loser * (0 - expected_loser)
        
    #     # Store historical data
    #     self.elo_history.append({
    #         'date': date,
    #         'event': event_name,
    #         'round': round_name,
    #         'player1': winner,
    #         'player2': loser,
    #         'score': f"{score[0]}-{score[1]}",
    #         'winner': winner,
    #         'player1_elo_before': winner_elo,
    #         'player2_elo_before': loser_elo,
    #         'player1_elo_after': winner_new_elo,
    #         'player2_elo_after': loser_new_elo
    #         # 'k_factor': k
    #     })
        
    #     # Update current ELOs
    #     self.player_elos[winner] = winner_new_elo
    #     self.player_elos[loser] = loser_new_elo

    def update_elo(self, winner: str, loser: str, event_name: str, 
               round_name: str, date: str, score: Tuple[int, int]) -> None:
        """OPTIMIZED VERSION with efficient game tracking."""
        # Initialize games tracking if needed
        if not hasattr(self, '_player_games'):
            self._player_games = {}
        
        # Get current ELOs and game counts
        winner_elo = self.get_player_elo(winner)
        loser_elo = self.get_player_elo(loser)
        winner_games = self._player_games.get(winner, 0)
        loser_games = self._player_games.get(loser, 0)
        
        # Get K-factors
        k_winner = self.k_factor_major if (winner_games < 30 and winner_elo < 2300) else self.k_factor
        k_loser = self.k_factor_major if (loser_games < 30 and loser_elo < 2300) else self.k_factor
        
        # Calculate expected scores
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
        
        # Update ELOs
        winner_new_elo = winner_elo + k_winner * (1 - expected_winner)
        loser_new_elo = loser_elo + k_loser * (0 - expected_loser)
        
        # Store historical data (minimal version)
        self.elo_history.append({
            'date': date, 'event': event_name, 'round': round_name,
            'player1': winner, 'player2': loser, 'score': f"{score[0]}-{score[1]}",
            'winner': winner, 'player1_elo_before': winner_elo, 'player2_elo_before': loser_elo,
            'player1_elo_after': winner_new_elo, 'player2_elo_after': loser_new_elo
        })
        
        # Update current ELOs and game counts
        self.player_elos[winner] = winner_new_elo
        self.player_elos[loser] = loser_new_elo
        self._player_games[winner] = winner_games + 1
        self._player_games[loser] = loser_games + 1

    def process_matches(self, matches_df: pd.DataFrame) -> None:
        """Process all matches in chronological order."""
        # Ensure dates are sorted
        matches_df = matches_df.copy()
        matches_df['EventDate'] = pd.to_datetime(matches_df['EventDate'], errors='coerce')
        matches_df = matches_df.sort_values('EventDate')
        
        for _, match in matches_df.iterrows():
            if (match['Winner'] and match['Winner'] != 'draw' and 
                match['Player1'] and match['Player2'] and
                pd.notna(match['Player1Score']) and pd.notna(match['Player2Score'])):
                
                if match['Winner'] == match['Player1']:
                    winner = match['Player1']
                    loser = match['Player2']
                    score = (match['Player1Score'], match['Player2Score'])
                else:
                    winner = match['Player2']
                    loser = match['Player1']
                    score = (match['Player2Score'], match['Player1Score'])
                
                self.update_elo(
                    winner=winner,
                    loser=loser,
                    event_name=match['Event'],
                    round_name=match['Round'],
                    date=match['EventDate'].strftime('%Y-%m-%d'),
                    score=score,
                )
    
    def get_elo_history_df(self) -> pd.DataFrame:
        """Get historical ELO data as DataFrame."""
        return pd.DataFrame(self.elo_history)
    
    def get_current_elos_df(self) -> pd.DataFrame:
        """Get current ELO ratings as DataFrame."""
        return pd.DataFrame([
            {'player': player, 'elo': elo} 
            for player, elo in self.player_elos.items()
        ]).sort_values('elo', ascending=False)
    
    def get_player_history(self, player_name: str) -> pd.DataFrame:
        """Get complete ELO history for a specific player."""
        history_df = self.get_elo_history_df()
        player_matches = history_df[
            (history_df['player1'] == player_name) | 
            (history_df['player2'] == player_name)
        ].copy()
        
        # Add player's ELO before and after each match
        def get_player_elo_before(row):
            return row['player1_elo_before'] if row['player1'] == player_name else row['player2_elo_before']
        
        def get_player_elo_after(row):
            return row['player1_elo_after'] if row['player1'] == player_name else row['player2_elo_after']
        
        def get_opponent(row):
            return row['player2'] if row['player1'] == player_name else row['player1']
        
        def get_result(row):
            return 'Win' if row['winner'] == player_name else 'Loss'
        
        player_matches['player_elo_before'] = player_matches.apply(get_player_elo_before, axis=1)
        player_matches['player_elo_after'] = player_matches.apply(get_player_elo_after, axis=1)
        player_matches['opponent'] = player_matches.apply(get_opponent, axis=1)
        player_matches['result'] = player_matches.apply(get_result, axis=1)
        player_matches['opponent_elo'] = player_matches.apply(
            lambda x: x['player2_elo_before'] if x['player1'] == player_name else x['player1_elo_before'], axis=1
        )
        
        return player_matches[[
            'date', 'event', 'round', 'opponent', 'result', 'score',
            'player_elo_before', 'player_elo_after', 'opponent_elo'
        ]].sort_values('date')

def load_all_match_data(csv_directory: str = '/Users/christopherharvey-hawes/Documents/Darts_Forecast/cleaned_darts_results/') -> pd.DataFrame:
    """Load all yearly CSV files into a single DataFrame."""
    csv_files = glob.glob(os.path.join(csv_directory, '*results_*.csv'))
    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)  # Track which year the data came from
            all_data.append(df)
            print(f"Loaded {len(df)} matches from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No CSV files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Clean and prepare data
    combined_df['EventDate'] = pd.to_datetime(combined_df['EventDate'], errors='coerce')
    combined_df = combined_df.dropna(subset=['EventDate'])
    combined_df = combined_df.sort_values('EventDate')
    
    print(f"Loaded {len(combined_df)} total matches")
    return combined_df

# def __get_games_played(matches_df: pd.DataFrame, player_name: str, current_date) -> int:
#     """Calculate total games played by a player before a match."""
#     event_date = current_date
#     date_filtered = matches_df[matches_df['EventDate'] < event_date]
#     player_matches = date_filtered[
#         (date_filtered['Player1'] == player_name) | (date_filtered['Player2'] == player_name)
#     ]
#     return len(player_matches)

def verify_chronological_order(matches_df: pd.DataFrame) -> bool:
    """Verify that matches are in proper chronological order."""
    # Check if dates are sorted
    dates_sorted = matches_df['EventDate'].is_monotonic_increasing
    if not dates_sorted:
        print("WARNING: Matches are NOT in chronological order!")
        
        # Find where the order breaks
        for i in range(1, len(matches_df)):
            if matches_df.iloc[i]['EventDate'] < matches_df.iloc[i-1]['EventDate']:
                print(f"  Order break at index {i}: {matches_df.iloc[i-1]['EventDate']} -> {matches_df.iloc[i]['EventDate']}")
                return False
    else:
        print("Matches are in proper chronological order")
        return True

def create_player_profiles(elo_calculator: DartsELOCalculator, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive player profiles with statistics."""
    player_stats = {}
    
    for player in elo_calculator.player_elos.keys():
        player_matches = matches_df[
            (matches_df['Player1'] == player) | (matches_df['Player2'] == player)
        ].copy()
        
        if len(player_matches) == 0:
            continue
        
        wins = 0
        losses = 0
        legs_won = 0
        legs_lost = 0
        tournaments_played = set()
        
        for _, match in player_matches.iterrows():
            tournaments_played.add(match['Event'])
            
            if match['Player1'] == player:
                player_score = match['Player1Score']
                opponent_score = match['Player2Score']
                if match['Winner'] == player:
                    wins += 1
                else:
                    losses += 1
            else:
                player_score = match['Player2Score']
                opponent_score = match['Player1Score']
                if match['Winner'] == player:
                    wins += 1
                else:
                    losses += 1
            
            if pd.notna(player_score):
                legs_won += player_score
            if pd.notna(opponent_score):
                legs_lost += opponent_score
        
        total_matches = wins + losses
        win_percentage = (wins / total_matches * 100) if total_matches > 0 else 0
        legs_ratio = (legs_won / legs_lost) if legs_lost > 0 else legs_won
        
        player_stats[player] = {
            'current_elo': elo_calculator.get_player_elo(player),
            'total_matches': total_matches,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_percentage,
            'legs_won': legs_won,
            'legs_lost': legs_lost,
            'legs_ratio': round(legs_ratio, 2),
            'tournaments_played': len(tournaments_played),
            'first_match': player_matches['EventDate'].min(),
            'last_match': player_matches['EventDate'].max()
        }
    
    profiles_df = pd.DataFrame.from_dict(player_stats, orient='index')
    profiles_df.index.name = 'player'
    profiles_df = profiles_df.reset_index()
    
    return profiles_df.sort_values('current_elo', ascending=False)

def main():
    """Main function to calculate ELO ratings and create player profiles."""
    print("Loading match data...")
    matches_df = load_all_match_data()
    print(f'loaded {len(matches_df)} matches from CSV files')
    verify_chronological_order(matches_df)

    print("Calculating ELO ratings...")
    elo_calculator = DartsELOCalculator()
    elo_calculator.process_matches(matches_df)
    
    print("Creating player profiles...")
    player_profiles = create_player_profiles(elo_calculator, matches_df)
    
    # Save results
    print("Saving results...")
    
    # Current ELO ratings
    current_elos = elo_calculator.get_current_elos_df()
    current_elos.to_csv('current_elo_ratings.csv', index=False)
    print(f"Saved current ELO ratings for {len(current_elos)} players")
    
    # Player profiles
    player_profiles.to_csv('player_profiles.csv', index=False)
    print(f"Saved profiles for {len(player_profiles)} players")
    
    # ELO history
    elo_history = elo_calculator.get_elo_history_df()
    elo_history.to_csv('elo_history.csv', index=False)
    print(f"Saved ELO history for {len(elo_history)} matches")
    
    # Top 10 players
    print("\n=== TOP 10 PLAYERS BY ELO ===")
    top_10 = current_elos.head(10)
    for i, (_, player) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {player['player']:30} ELO: {player['elo']:.0f}")
    
    # Example: Get history for top player
    if len(top_10) > 0:
        top_player = top_10.iloc[0]['player']
        print(f"\n=== RECENT HISTORY FOR {top_player} ===")
        player_history = elo_calculator.get_player_history(top_player)
        recent_matches = player_history.tail(5)
        for _, match in recent_matches.iterrows():
            print(f"{match['date']}: {match['result']} vs {match['opponent']} {match['score']} "
                  f"(ELO: {match['player_elo_before']:.0f} → {match['player_elo_after']:.0f})")

if __name__ == "__main__":
    main()