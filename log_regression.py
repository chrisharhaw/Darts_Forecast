import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple
from tqdm import tqdm 
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class DartsRatingCalculator:
    def __init__(self, initial_rating: float = 0.0, learning_rate: float = 0.1, 
                 regularization: float = 1.0, min_games_for_stable: int = 30):
        """
        Initialize rating calculator using logistic regression approach.
        
        Args:
            initial_rating: Starting rating for new players
            learning_rate: How quickly ratings update (similar to K-factor)
            regularization: Regularization strength for logistic regression
            min_games_for_stable: Minimum games before considering a player's rating stable
        """
        self.initial_rating = initial_rating
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.min_games_for_stable = min_games_for_stable
        
        self.player_ratings = {}  # Current player ratings
        self.player_games = {}    # Games played per player
        self.rating_history = []  # Historical rating data
        self.feature_matrix = []  # For batch updates if desired
        self.target_vector = []   # For batch updates if desired
    
    def get_player_rating(self, player_name: str) -> float:
        """Get current rating for a player, initialize if new."""
        if player_name not in self.player_ratings:
            self.player_ratings[player_name] = self.initial_rating
            self.player_games[player_name] = 0
        return self.player_ratings[player_name]
    
    def get_player_games(self, player_name: str) -> int:
        """Get number of games played by a player."""
        return self.player_games.get(player_name, 0)
    
    def calculate_win_probability(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate probability that player A beats player B using logistic function.
        This is equivalent to expected_score in ELO but more statistically sound.
        """
        return 1 / (1 + np.exp(-(rating_a - rating_b)))
    
    def update_ratings(self, winner: str, loser: str, event_name: str, 
                            round_name: str, date: str, score: Tuple[int, int]) -> None:
        """
        Update ratings using gradient descent (similar to ELO but with proper probabilities).
        """
        # Get current ratings and game counts
        winner_rating = self.get_player_rating(winner)
        loser_rating = self.get_player_rating(loser)
        winner_games = self.get_player_games(winner)
        loser_games = self.get_player_games(loser)
        
        # Calculate win probability
        win_prob = self.calculate_win_probability(winner_rating, loser_rating)
        
        # Calculate gradient (error between actual and predicted)
        # Actual result: winner wins (1), loser loses (0)
        error_winner = 1 - win_prob  # Winner should have won with probability 1
        error_loser = 0 - (1 - win_prob)  # = -(1 - win_prob)
        
        # Adaptive learning rate - higher for new players
        winner_lr = self.learning_rate * (2.0 if winner_games < self.min_games_for_stable else 1.0)
        loser_lr = self.learning_rate * (2.0 if loser_games < self.min_games_for_stable else 1.0)
        
        # Update ratings using gradient descent
        winner_new_rating = winner_rating + winner_lr * error_winner
        loser_new_rating = loser_rating + loser_lr * error_loser
        
        # Store historical data
        self.rating_history.append({
            'date': date,
            'event': event_name,
            'round': round_name,
            'player1': winner,
            'player2': loser,
            'score': f"{score[0]}-{score[1]}",
            'winner': winner,
            'player1_rating_before': winner_rating,
            'player2_rating_before': loser_rating,
            'player1_rating_after': winner_new_rating,
            'player2_rating_after': loser_new_rating,
            'win_probability': win_prob,
            'player1_games_before': winner_games,
            'player2_games_before': loser_games
        })
        
        # Update current ratings and game counts
        self.player_ratings[winner] = winner_new_rating
        self.player_ratings[loser] = loser_new_rating
        self.player_games[winner] = winner_games + 1
        self.player_games[loser] = loser_games + 1
    
    def fit_batch_logistic_regression(self, matches_df: pd.DataFrame) -> None:
        """
        Alternative: Fit logistic regression on all historical data at once.
        This gives more stable ratings but loses the time-series aspect.
        """
        print("Fitting batch logistic regression...")
        
        # Prepare features (player indicators) and target (winner)
        player_ids = {}
        feature_rows = []
        target_rows = []
        
        # Create player mapping
        all_players = set(matches_df['Player1']).union(set(matches_df['Player2']))
        player_ids = {player: idx for idx, player in enumerate(all_players)}
        n_players = len(player_ids)
        
        for _, match in matches_df.iterrows():
            if match['Winner'] not in ['draw', ''] and pd.notna(match['Winner']):
                # Create feature vector: +1 for player1, -1 for player2
                features = np.zeros(n_players)
                features[player_ids[match['Player1']]] = 1
                features[player_ids[match['Player2']]] = -1
                
                feature_rows.append(features)
                
                # Target: 1 if player1 wins, 0 if player2 wins
                target = 1 if match['Winner'] == match['Player1'] else 0
                target_rows.append(target)
        
        # Fit logistic regression
        X = np.array(feature_rows)
        y = np.array(target_rows)
        
        model = LogisticRegression(C=self.regularization, fit_intercept=False)
        model.fit(X, y)
        
        # Extract player ratings from coefficients
        ratings = model.coef_[0]
        for player, idx in player_ids.items():
            self.player_ratings[player] = ratings[idx]
            self.player_games[player] = len(matches_df[
                (matches_df['Player1'] == player) | (matches_df['Player2'] == player)
            ])
        
        print(f"Fitted logistic regression on {len(X)} matches")
    
    def process_matches(self, matches_df: pd.DataFrame) -> None:
        """Process matches in chronological order."""
        matches_df = matches_df.copy()
        matches_df['EventDate'] = pd.to_datetime(matches_df['EventDate'], errors='coerce')
        matches_df = matches_df.sort_values('EventDate')
        
        # valid_matches = matches_df[
        #     (matches_df['Winner'].notna()) & 
        #     (matches_df['Winner'] != 'draw') &
        #     (matches_df['Player1'].notna()) & 
        #     (matches_df['Player2'].notna())
        # ]
        valid_matches = matches_df
        print(f"Processing {len(valid_matches)} valid matches with online updates...")
        
        for idx, match in tqdm(valid_matches.iterrows(), total=len(valid_matches), desc="Updating ratings"):
            if match['Winner'] == match['Player1']:
                winner = match['Player1']
                loser = match['Player2']
                score = (match['Player1Score'], match['Player2Score'])
            else:
                winner = match['Player2']
                loser = match['Player1']
                score = (match['Player2Score'], match['Player1Score'])
            
            self.update_ratings(
                winner=winner, loser=loser,
                event_name=match['Event'], round_name=match['Round'],
                date=match['EventDate'].strftime('%Y-%m-%d'), score=score
            )
    
    def get_rating_history_df(self) -> pd.DataFrame:
        """Get historical rating data as DataFrame."""
        return pd.DataFrame(self.rating_history)
    
    def get_current_ratings_df(self) -> pd.DataFrame:
        """Get current ratings as DataFrame."""
        ratings_df = pd.DataFrame([
            {
                'player': player, 
                'rating': rating,
                'games_played': self.player_games.get(player, 0),
                'win_prob_vs_avg': self.calculate_win_probability(rating, 0)  # vs average player
            } 
            for player, rating in self.player_ratings.items()
        ]).sort_values('rating', ascending=False)
        
        # Convert ratings to more interpretable scale (similar to ELO)
        ratings_df['elo_equivalent'] = ratings_df['rating'] * 400 / np.log(10) + 1500
        return ratings_df
    
    def get_player_win_probability(self, player1: str, player2: str) -> float:
        """Get probability that player1 beats player2."""
        rating1 = self.get_player_rating(player1)
        rating2 = self.get_player_rating(player2)
        return self.calculate_win_probability(rating1, rating2)
    
    def get_player_history(self, player_name: str) -> pd.DataFrame:
        """Get complete rating history for a specific player."""
        history_df = self.get_rating_history_df()
        player_matches = history_df[
            (history_df['player1'] == player_name) | 
            (history_df['player2'] == player_name)
        ].copy()
        
        def get_player_rating_before(row):
            return row['player1_rating_before'] if row['player1'] == player_name else row['player2_rating_before']
        
        def get_player_rating_after(row):
            return row['player1_rating_after'] if row['player1'] == player_name else row['player2_rating_after']
        
        def get_opponent(row):
            return row['player2'] if row['player1'] == player_name else row['player1']
        
        def get_result(row):
            return 'Win' if row['winner'] == player_name else 'Loss'
        
        def get_opponent_rating(row):
            return row['player2_rating_before'] if row['player1'] == player_name else row['player1_rating_before']
        
        def get_expected_win_prob(row):
            rating_before = get_player_rating_before(row)
            opponent_rating = get_opponent_rating(row)
            return self.calculate_win_probability(rating_before, opponent_rating)
        
        player_matches['player_rating_before'] = player_matches.apply(get_player_rating_before, axis=1)
        player_matches['player_rating_after'] = player_matches.apply(get_player_rating_after, axis=1)
        player_matches['opponent'] = player_matches.apply(get_opponent, axis=1)
        player_matches['result'] = player_matches.apply(get_result, axis=1)
        player_matches['opponent_rating'] = player_matches.apply(get_opponent_rating, axis=1)
        player_matches['expected_win_prob'] = player_matches.apply(get_expected_win_prob, axis=1)
        
        return player_matches[[
            'date', 'event', 'round', 'opponent', 'result', 'score',
            'player_rating_before', 'player_rating_after', 'opponent_rating',
            'expected_win_prob', 'win_probability'
        ]].sort_values('date')

# Keep your existing helper functions but update create_player_profiles
def create_player_profiles(rating_calculator: DartsRatingCalculator, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive player profiles with statistics."""
    player_stats = {}
    
    all_players = list(rating_calculator.player_ratings.keys())
    
    for player in tqdm(all_players, desc="Processing players"):
        player_matches = matches_df[
            (matches_df['Player1'] == player) | (matches_df['Player2'] == player)
        ].copy()
        
        if len(player_matches) == 0:
            continue
        
        # Vectorized calculations
        is_player1 = player_matches['Player1'] == player
        player_scores = np.where(is_player1, 
                               player_matches['Player1Score'], 
                               player_matches['Player2Score'])
        opponent_scores = np.where(is_player1,
                                 player_matches['Player2Score'],
                                 player_matches['Player1Score'])
        
        wins = np.where(is_player1,
                       player_matches['Winner'] == player_matches['Player1'],
                       player_matches['Winner'] == player_matches['Player2']).sum()
        
        losses = len(player_matches) - wins
        legs_won = player_scores.sum()
        legs_lost = opponent_scores.sum()
        
        total_matches = wins + losses
        win_percentage = (wins / total_matches * 100) if total_matches > 0 else 0
        legs_ratio = (legs_won / legs_lost) if legs_lost > 0 else legs_won
        
        player_stats[player] = {
            'rating': rating_calculator.get_player_rating(player),
            'elo_equivalent': rating_calculator.get_player_rating(player) * 400 / np.log(10) + 1500,
            'total_matches': total_matches,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_percentage,
            'legs_won': legs_won,
            'legs_lost': legs_lost,
            'legs_ratio': round(legs_ratio, 2),
            'tournaments_played': player_matches['Event'].nunique(),
            'first_match': player_matches['EventDate'].min(),
            'last_match': player_matches['EventDate'].max(),
            'games_played': rating_calculator.get_player_games(player)
        }
    
    profiles_df = pd.DataFrame.from_dict(player_stats, orient='index')
    profiles_df.index.name = 'player'
    profiles_df = profiles_df.reset_index()
    
    return profiles_df.sort_values('rating', ascending=False)

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

def main():
    """Main function to calculate ratings using logistic regression approach."""
    print("Loading match data...")
    matches_df = load_all_match_data()
    print(f'Loaded {len(matches_df)} matches from CSV files')
    verify_chronological_order(matches_df)

    print("Calculating ratings using logistic regression approach...")
    
    # Option 1: Online updates (maintains time series)
    rating_calculator = DartsRatingCalculator(learning_rate=0.1, regularization=10.0, min_games_for_stable=30)
    rating_calculator.process_matches(matches_df)
    
    # Option 2: Batch logistic regression (uncomment to use)
    # rating_calculator.fit_batch_logistic_regression(matches_df)
    
    print("Creating player profiles...")
    player_profiles = create_player_profiles(rating_calculator, matches_df)
    
    # Save results
    print("Saving results...")
    
    # Current ratings
    current_ratings = rating_calculator.get_current_ratings_df()
    current_ratings.to_csv('current_ratings_logistic.csv', index=False)
    print(f"Saved current ratings for {len(current_ratings)} players")
    
    # Player profiles
    player_profiles.to_csv('player_profiles_logistic.csv', index=False)
    print(f"Saved profiles for {len(player_profiles)} players")
    
    # Rating history
    rating_history = rating_calculator.get_rating_history_df()
    rating_history.to_csv('rating_history_logistic.csv', index=False)
    print(f"Saved rating history for {len(rating_history)} matches")
    
    # Top 10 players
    print("\n=== TOP 10 PLAYERS BY RATING ===")
    top_10 = current_ratings.head(10)
    for i, (_, player) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {player['player']:30} Rating: {player['rating']:.3f} "
              f"(ELO equiv: {player['elo_equivalent']:.0f})")
    
    # Example prediction
    if len(top_10) > 1:
        player1 = top_10.iloc[0]['player']
        player2 = top_10.iloc[1]['player']
        win_prob = rating_calculator.get_player_win_probability(player1, player2)
        print(f"\nPrediction: {player1} vs {player2}")
        print(f"Win probability for {player1}: {win_prob:.1%}")
        print(f"Win probability for {player2}: {(1-win_prob):.1%}")

# Keep your existing load_all_match_data and verify_chronological_order functions

if __name__ == "__main__":
    main()