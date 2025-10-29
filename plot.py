import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from elo_calc import DartsELOCalculator, load_all_match_data

def plot_elo_progression(elo_calculator, player_names: list[str], save_path: str = 'elo_progression.png'):
    """Plot ELO progression for specific players over time."""
    plt.figure(figsize=(12, 8))
    
    for player in player_names:
        player_history = elo_calculator.get_player_history(player)
        if len(player_history) > 0:
            # Convert dates for plotting
            player_history['date'] = pd.to_datetime(player_history['date'])
            player_history = player_history.sort_values('date')
            
            # Plot ELO progression
            plt.plot(player_history['date'], player_history['player_elo_after'], 
                    label=player, linewidth=2, marker='o', markersize=3)
    
    plt.title('ELO Rating Progression Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('ELO Rating', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_players_elo(elo_calculator, top_n: int = 20, save_path: str = 'top_players_elo.png'):
    """Create a bar chart of top players by ELO."""
    current_elos = elo_calculator.get_current_elos_df()
    top_players = current_elos.head(top_n)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(top_players)), top_players['elo'], color='skyblue', edgecolor='navy')
    
    plt.gca().invert_yaxis()  # Highest ELO at top
    plt.title(f'Top {top_n} Players by ELO Rating', fontsize=16, fontweight='bold')
    plt.xlabel('ELO Rating', fontsize=12)
    plt.yticks(range(len(top_players)), top_players['player'], fontsize=10)
    
    # Add ELO values on bars
    for i, (bar, elo) in enumerate(zip(bars, top_players['elo'])):
        plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{elo:.0f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Usage example:
if __name__ == "__main__":
    # Load your data and calculate ELO first
    matches_df = load_all_match_data()
    elo_calculator = DartsELOCalculator()
    elo_calculator.process_matches(matches_df)
    
    # Get top 5 players for plotting
    top_players = elo_calculator.get_current_elos_df().head(5)['player'].tolist()
    
    # Create plots
    plot_elo_progression(elo_calculator, top_players)
    plot_top_players_elo(elo_calculator, top_n=20)