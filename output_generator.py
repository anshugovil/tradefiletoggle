"""
Output Generator Module
Handles creation and export of all output files
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generates and saves all output files"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_all_outputs(self, 
                        parsed_trades_df: pd.DataFrame,
                        starting_positions_df: pd.DataFrame,
                        processed_trades_df: pd.DataFrame,
                        final_positions_df: pd.DataFrame,
                        file_prefix: str = "output") -> Dict[str, Path]:
        """
        Save all four output files
        Returns dictionary of file type to file path
        """
        output_files = {}
        
        # File 1: Parsed Trade File (original trades from parser)
        parsed_trades_file = self.output_dir / f"{file_prefix}_1_parsed_trades_{self.timestamp}.csv"
        parsed_trades_df.to_csv(parsed_trades_file, index=False)
        output_files['parsed_trades'] = parsed_trades_file
        logger.info(f"Saved parsed trades to {parsed_trades_file}")
        
        # File 2: Starting Position File
        starting_pos_file = self.output_dir / f"{file_prefix}_2_starting_positions_{self.timestamp}.csv"
        starting_positions_df.to_csv(starting_pos_file, index=False)
        output_files['starting_positions'] = starting_pos_file
        logger.info(f"Saved starting positions to {starting_pos_file}")
        
        # File 3: Processed Trade File (main output with strategies)
        processed_trades_file = self.output_dir / f"{file_prefix}_3_processed_trades_{self.timestamp}.csv"
        processed_trades_df.to_csv(processed_trades_file, index=False)
        output_files['processed_trades'] = processed_trades_file
        logger.info(f"Saved processed trades to {processed_trades_file}")
        
        # Also save as Excel for better readability
        try:
            processed_trades_excel = self.output_dir / f"{file_prefix}_3_processed_trades_{self.timestamp}.xlsx"
            with pd.ExcelWriter(processed_trades_excel, engine='openpyxl') as writer:
                processed_trades_df.to_excel(writer, sheet_name='Processed Trades', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['Processed Trades']
                for idx, column in enumerate(processed_trades_df.columns):
                    max_length = max(
                        processed_trades_df[column].astype(str).map(len).max(),
                        len(str(column))
                    )
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[chr(65 + idx % 26)].width = adjusted_width
            
            output_files['processed_trades_excel'] = processed_trades_excel
            logger.info(f"Saved processed trades Excel to {processed_trades_excel}")
        except Exception as e:
            logger.warning(f"Could not save Excel file: {e}")
        
        # File 4: Final Position File
        final_pos_file = self.output_dir / f"{file_prefix}_4_final_positions_{self.timestamp}.csv"
        final_positions_df.to_csv(final_pos_file, index=False)
        output_files['final_positions'] = final_pos_file
        logger.info(f"Saved final positions to {final_pos_file}")
        
        # Create summary report
        summary_file = self._create_summary_report(
            parsed_trades_df, 
            starting_positions_df, 
            processed_trades_df, 
            final_positions_df
        )
        output_files['summary'] = summary_file
        
        return output_files
    
    def _create_summary_report(self,
                              parsed_trades_df: pd.DataFrame,
                              starting_positions_df: pd.DataFrame,
                              processed_trades_df: pd.DataFrame,
                              final_positions_df: pd.DataFrame) -> Path:
        """Create a summary report of the processing"""
        summary_file = self.output_dir / f"summary_report_{self.timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRADE PROCESSING SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Starting positions summary
            f.write("STARTING POSITIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total positions: {len(starting_positions_df)}\n")
            if len(starting_positions_df) > 0:
                f.write(f"Long positions: {len(starting_positions_df[starting_positions_df['QTY'] > 0])}\n")
                f.write(f"Short positions: {len(starting_positions_df[starting_positions_df['QTY'] < 0])}\n")
            f.write("\n")
            
            # Trades summary
            f.write("TRADES PROCESSED:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total trades: {len(parsed_trades_df)}\n")
            f.write(f"Trades after processing: {len(processed_trades_df)}\n")
            
            # Split trades
            if 'Split?' in processed_trades_df.columns:
                split_trades = processed_trades_df[processed_trades_df['Split?'] == 'Yes']
                f.write(f"Split trades: {len(split_trades)} (from {len(split_trades)//2} original trades)\n")
            
            # Opposite trades
            if 'Opposite?' in processed_trades_df.columns:
                opposite_trades = processed_trades_df[processed_trades_df['Opposite?'] == 'Yes']
                f.write(f"Trades with opposite strategy: {len(opposite_trades)}\n")
            
            # Strategy breakdown
            if 'Strategy' in processed_trades_df.columns:
                f.write("\nStrategy Breakdown:\n")
                strategy_counts = processed_trades_df['Strategy'].value_counts()
                for strategy, count in strategy_counts.items():
                    f.write(f"  {strategy}: {count} trades\n")
            
            f.write("\n")
            
            # Final positions summary
            f.write("FINAL POSITIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total positions: {len(final_positions_df)}\n")
            if len(final_positions_df) > 0:
                f.write(f"Long positions: {len(final_positions_df[final_positions_df['QTY'] > 0])}\n")
                f.write(f"Short positions: {len(final_positions_df[final_positions_df['QTY'] < 0])}\n")
                
                # Position changes
                f.write("\nPosition Changes:\n")
                initial_tickers = set(starting_positions_df['Ticker'].unique()) if len(starting_positions_df) > 0 else set()
                final_tickers = set(final_positions_df['Ticker'].unique()) if len(final_positions_df) > 0 else set()
                
                new_positions = final_tickers - initial_tickers
                closed_positions = initial_tickers - final_tickers
                
                if new_positions:
                    f.write(f"  New positions opened: {len(new_positions)}\n")
                if closed_positions:
                    f.write(f"  Positions closed: {len(closed_positions)}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Created summary report at {summary_file}")
        return summary_file
    
    def create_trade_dataframe_from_positions(self, positions: List) -> pd.DataFrame:
        """Convert Position objects to DataFrame for parsed trades output"""
        trades_data = []
        
        for pos in positions:
            trade_dict = {
                'Symbol': pos.symbol,
                'Bloomberg_Ticker': pos.bloomberg_ticker,
                'Expiry': pos.expiry_date,
                'Strike': pos.strike_price,
                'Security_Type': pos.security_type,
                'Lots': pos.position_lots,
                'Lot_Size': pos.lot_size,
                'Quantity': pos.position_lots * pos.lot_size
            }
            trades_data.append(trade_dict)
        
        return pd.DataFrame(trades_data)
