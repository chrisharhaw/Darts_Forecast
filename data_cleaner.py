import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime


class DartsDataCleaner:
    def __init__(self):
        self.cleaning_stats = {}
    
    def load_all_data(self, csv_directory: str = '/Users/christopherharvey-hawes/Documents/Darts_Forecast/darts_results/', file_pattern: str = 'darts_results_*.csv') -> pd.DataFrame:
        """Load all yearly CSV files into a single DataFrame."""
        csv_files = glob.glob(os.path.join(csv_directory, file_pattern))
        all_data = []
        
        print("Loading CSV files...")
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = os.path.basename(file)
                all_data.append(df)
                print(f"  ✓ Loaded {len(df):>4} matches from {file}")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if not all_data:
            raise ValueError("No CSV files found!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert date column
        combined_df['EventDate'] = pd.to_datetime(combined_df['EventDate'], dayfirst=True, errors='coerce')
        
        print(f"✓ Successfully loaded {len(combined_df)} total matches from {len(csv_files)} files")
        return combined_df
    
    def handle_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle matches with missing or invalid dates more carefully."""
        initial_count = len(df)
        
        # Remove rows with missing/empty dates
        missing_dates = df['EventDate'].isna() | (df['EventDate'] == '')
        cleaned_df = df[~missing_dates].copy()
        
        removed_count = initial_count - len(cleaned_df)
        self.cleaning_stats['missing_dates_removed'] = removed_count
        self.cleaning_stats['after_date_removal'] = len(cleaned_df)
        
        print(f"  Removed {removed_count} matches with missing dates")
        return cleaned_df
    
    def remove_scoreless_draws(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove matches where both players have 0 score (scoreless draws)."""
        initial_count = len(df)
        
        # Identify scoreless draws
        scoreless_zero = (
            (df['Player1Score'] == 0) & 
            (df['Player2Score'] == 0) &
            (df['Player1Score'].notna()) & 
            (df['Player2Score'].notna())
        )

        empty_scores = (
            (df['Player1Score'].isna() | (df['Player1Score'].astype(str).str.strip() == '')) &
            (df['Player2Score'].isna() | (df['Player2Score'].astype(str).str.strip() == ''))
        )

        scoreless_mask = scoreless_zero | empty_scores
        
        scoreless_count = scoreless_mask.sum()
        cleaned_df = df[~scoreless_mask].copy()
        
        self.cleaning_stats['scoreless_draws_removed'] = scoreless_count
        self.cleaning_stats['after_scoreless_removal'] = len(cleaned_df)
        
        print(f"  Removed {scoreless_count} scoreless draws")
        return cleaned_df
    
    def remove_invalid_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove matches with invalid or impossible scores."""
        initial_count = len(df)
        
        # Remove matches where both scores are NaN
        both_scores_nan = df['Player1Score'].isna() & df['Player2Score'].isna()
        
        # Remove matches where scores are negative
        negative_scores = (df['Player1Score'] < 0) | (df['Player2Score'] < 0)
        
        # Remove matches where both scores are 0 (already handled, but just in case)
        both_zero = (df['Player1Score'] == 0) & (df['Player2Score'] == 0)
        
        # Remove matches where winner doesn't match scores
        winner_mismatch = (
            ((df['Winner'] == df['Player1']) & (df['Player1Score'] <= df['Player2Score'])) |
            ((df['Winner'] == df['Player2']) & (df['Player2Score'] <= df['Player1Score'])) |
            ((df['Winner'] == 'draw') & (df['Player1Score'] != df['Player2Score']))
        )
        
        invalid_mask = both_scores_nan | negative_scores | both_zero | winner_mismatch
        cleaned_df = df[~invalid_mask].copy()
        
        self.cleaning_stats['invalid_scores_removed'] = initial_count - len(cleaned_df)
        self.cleaning_stats['after_invalid_removal'] = len(cleaned_df)
        
        print(f"  Removed {initial_count - len(cleaned_df)} matches with invalid scores")
        return cleaned_df
    
    def clean_player_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and clean player names."""
        cleaned_df = df.copy()
        
        # Remove extra whitespace
        cleaned_df['Player1'] = cleaned_df['Player1'].str.strip()
        cleaned_df['Player2'] = cleaned_df['Player2'].str.strip()
        cleaned_df['Winner'] = cleaned_df['Winner'].str.strip()
        
        # Replace empty strings with NaN
        cleaned_df['Player1'] = cleaned_df['Player1'].replace('', np.nan)
        cleaned_df['Player2'] = cleaned_df['Player2'].replace('', np.nan)
        cleaned_df['Winner'] = cleaned_df['Winner'].replace('', np.nan)
        
        # Remove matches with missing player names
        initial_count = len(cleaned_df)
        missing_players = cleaned_df['Player1'].isna() | cleaned_df['Player2'].isna()
        cleaned_df = cleaned_df[~missing_players]
        
        self.cleaning_stats['missing_players_removed'] = initial_count - len(cleaned_df)
        self.cleaning_stats['after_player_cleaning'] = len(cleaned_df)
        
        print(f"  Removed {initial_count - len(cleaned_df)} matches with missing player names")
        return cleaned_df
    
    def remove_duplicate_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate matches based on key columns."""
        initial_count = len(df)
        
        # Create a unique match identifier
        df_with_id = df.copy()
        df_with_id['match_id'] = (
            df['Event'] + '|' + 
            df['EventDate'].astype(str) + '|' + 
            df['Player1'] + '|' + 
            df['Player2'] + '|' + 
            df['Player1Score'].astype(str) + '|' + 
            df['Player2Score'].astype(str)
        )
        
        # Remove duplicates, keeping the first occurrence
        cleaned_df = df_with_id.drop_duplicates(subset=['match_id'], keep='first')
        cleaned_df = cleaned_df.drop('match_id', axis=1)
        
        self.cleaning_stats['duplicates_removed'] = initial_count - len(cleaned_df)
        self.cleaning_stats['after_duplicate_removal'] = len(cleaned_df)
        
        print(f"  Removed {initial_count - len(cleaned_df)} duplicate matches")
        return cleaned_df
    
    def validate_winners(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure winner field matches the actual scores."""
        cleaned_df = df.copy()
        
        # Recalculate winner based on scores to fix any inconsistencies
        def determine_winner(row):
            if pd.isna(row['Player1Score']) or pd.isna(row['Player2Score']):
                return row['Winner']  # Keep original if scores missing
            
            if row['Player1Score'] > row['Player2Score']:
                return row['Player1']
            elif row['Player2Score'] > row['Player1Score']:
                return row['Player2']
            else:
                return 'draw'
        
        cleaned_df['Winner'] = cleaned_df.apply(determine_winner, axis=1)
        
        # Count corrected winners
        original_winners = df['Winner']
        new_winners = cleaned_df['Winner']
        corrections = (original_winners != new_winners).sum()
        
        self.cleaning_stats['winner_corrections'] = corrections
        print(f"  Corrected {corrections} winner assignments")
        
        return cleaned_df
    
    def clean_event_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize event names."""
        cleaned_df = df.copy()
        
        # Remove extra whitespace and standardize case
        cleaned_df['Event'] = cleaned_df['Event'].str.strip()
        cleaned_df['Round'] = cleaned_df['Round'].str.strip()
        
        # Replace empty event names with "Unknown Event"
        cleaned_df['Event'] = cleaned_df['Event'].replace('', 'Unknown Event')
        cleaned_df['Event'] = cleaned_df['Event'].fillna('Unknown Event')
        
        print("  Standardized event and round names")
        return cleaned_df
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """Generate a report of all cleaning operations."""
        if not self.cleaning_stats:
            return pd.DataFrame()
        
        report_data = []
        stages = [
            ('initial_count', 'Initial dataset'),
            ('dates_parsed', 'Dates successfully parsed'),
            ('after_scoreless_removal', 'After removing scoreless draws'),
            ('after_invalid_removal', 'After removing invalid scores'),
            ('after_player_cleaning', 'After cleaning player names'),
            ('after_duplicate_removal', 'After removing duplicates'),
            ('after_date_cleaning', 'After handling missing dates')
        ]
        
        for stat_key, description in stages:
            if stat_key in self.cleaning_stats:
                report_data.append({
                    'Stage': description,
                    'Matches': self.cleaning_stats[stat_key]
                })
        
        return pd.DataFrame(report_data)
    
    def clean_dataset(self, csv_directory: str = '/Users/christopherharvey-hawes/Documents/Darts_Forecast/darts_results/', 
                      output_directory: str = '/Users/christopherharvey-hawes/Documents/Darts_Forecast/cleaned_darts_results/',
                        save_cleaned: bool = True) -> pd.DataFrame:
        """
        Complete data cleaning pipeline.
        
        Args:
            csv_directory: Directory containing CSV files
            save_cleaned: Whether to save cleaned data to new CSV files
        
        Returns:
            Cleaned DataFrame
        """
        print("Starting data cleaning pipeline...")
        
        # Load data
        df = self.load_all_data(csv_directory)
        self.cleaning_stats['initial_count'] = len(df)
        
        # Apply cleaning steps
        df = self.remove_scoreless_draws(df)
        df = self.remove_invalid_scores(df)
        df = self.clean_player_names(df)
        df = self.remove_duplicate_matches(df)
        df = self.handle_missing_dates(df)
        df = self.validate_winners(df)
        df = self.clean_event_names(df)
        
        # Sort by date
        df = df.sort_values('EventDate').reset_index(drop=True)
        
        # Generate report
        report = self.get_cleaning_report()
        print("\n" + "="*50)
        print("CLEANING REPORT")
        print("="*50)
        for _, row in report.iterrows():
            print(f"{row['Stage']:<40} {row['Matches']:>6} matches")
        
        total_removed = self.cleaning_stats['initial_count'] - len(df)
        removal_percentage = (total_removed / self.cleaning_stats['initial_count']) * 100
        
        print(f"\nTotal matches removed: {total_removed} ({removal_percentage:.1f}%)")
        print(f"Final dataset: {len(df)} matches")
        
        # Save cleaned data
        if save_cleaned:
            self.save_cleaned_data(df, output_directory)
        
        return df
    
    def save_cleaned_data(self, df: pd.DataFrame, output_directory: str = '/Users/christopherharvey-hawes/Documents/Darts_Forecast/cleaned_darts_results/'):
        """Save cleaned data, preserving the yearly file structure."""
        # Save complete cleaned dataset
        output_file = os.path.join(output_directory, 'darts_matches_cleaned.csv')
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved complete cleaned dataset to {output_file}")
        
        # Also save by year to maintain structure
        for source_file in df['source_file'].unique():
            year_data = df[df['source_file'] == source_file].copy()
            year_data = year_data.drop('source_file', axis=1)
            
            # Create cleaned filename
            clean_filename = source_file.replace('.csv', '_clean.csv')
            clean_path = os.path.join(output_directory, clean_filename)
            
            year_data.to_csv(clean_path, index=False)
            print(f"✓ Saved {len(year_data)} matches to {clean_filename}")
        
        # Save cleaning report
        report = self.get_cleaning_report()
        report_path = os.path.join(output_directory, 'cleaning_report.csv')
        report.to_csv(report_path, index=False)
        print(f"✓ Saved cleaning report to {report_path}")

def quick_clean_existing_files():
    """
    Quick function to clean all existing result files in place.
    This modifies your original yearly files - use with caution!
    """
    cleaner = DartsDataCleaner()
    
    # Load and clean all data
    cleaned_df = cleaner.clean_dataset(save_cleaned=True)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    cleaner = DartsDataCleaner()
    
    # Clean the dataset (this will create new cleaned files)
    cleaned_data = cleaner.clean_dataset(save_cleaned=True)
    
    # You can also use the quick function to clean in place
    # cleaned_data = quick_clean_existing_files()
    
    print(f"\n🎯 Data cleaning complete!")
    print(f"📊 Final dataset contains {len(cleaned_data)} matches")
    
    # Show a sample of the cleaned data
    print("\nSample of cleaned data:")
    print(cleaned_data[['Event', 'EventDate', 'Player1', 'Player2', 'Player1Score', 'Player2Score']].head())