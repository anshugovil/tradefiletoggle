"""
Output Generator Module - ENHANCED VERSION
Handles creation and export of all output files including missing mappings report
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generates and saves all output files"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.missing_mappings = {'positions': [], 'trades': []}
        
    def save_all_outputs(self, 
                        parsed_trades_df: pd.DataFrame,
                        starting_positions_df: pd.DataFrame,
                        processed_trades_df: pd.DataFrame,
                        final_positions_df: pd.DataFrame,
                        file_prefix: str = "output",
                        input_parser=None,
                        trade_parser=None) -> Dict[str, Path]:
        """
        Save all output files including missing mappings report
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
        
        # File 5: Missing Mappings Report
        if input_parser or trade_parser:
            missing_mappings_file = self.create_missing_mappings_report(input_parser, trade_parser)
            if missing_mappings_file:
                output_files['missing_mappings'] = missing_mappings_file
        
        # Create summary report
        summary_file = self._create_summary_report(
            parsed_trades_df, 
            starting_positions_df, 
            processed_trades_df, 
            final_positions_df,
            input_parser,
            trade_parser
        )
        output_files['summary'] = summary_file
        
        return output_files
    
    def create_missing_mappings_report(self, input_parser=None, trade_parser=None) -> Optional[Path]:
        """
        Create a report of all unmapped symbols from both parsers
        Returns the path to the CSV file
        """
        all_missing = []
        
        # Collect from input parser (positions)
        if input_parser and hasattr(input_parser, 'unmapped_symbols'):
            for item in input_parser.unmapped_symbols:
                all_missing.append({
                    'Source': 'Position File',
                    'Symbol': item.get('symbol', ''),
                    'Expiry': item.get('expiry', ''),
                    'Quantity': item.get('position_lots', 0),
                    'Suggested_Ticker': self._suggest_ticker(item.get('symbol', '')),
                    'Underlying': '',
                    'Exchange': '',
                    'Lot_Size': ''
                })
        
        # Collect from trade parser
        if trade_parser and hasattr(trade_parser, 'unmapped_symbols'):
            for item in trade_parser.unmapped_symbols:
                all_missing.append({
                    'Source': 'Trade File',
                    'Symbol': item.get('symbol', ''),
                    'Expiry': item.get('expiry', ''),
                    'Quantity': item.get('position_lots', 0),
                    'Suggested_Ticker': self._suggest_ticker(item.get('symbol', '')),
                    'Underlying': '',
                    'Exchange': '',
                    'Lot_Size': ''
                })
        
        if not all_missing:
            logger.info("No missing mappings found")
            return None
        
        # Create DataFrame and remove duplicates
        df = pd.DataFrame(all_missing)
        
        # Group by symbol to consolidate
        unique_symbols = df.groupby('Symbol').agg({
            'Source': lambda x: ', '.join(sorted(set(x))),
            'Expiry': 'first',
            'Quantity': 'sum',
            'Suggested_Ticker': 'first',
            'Underlying': 'first',
            'Exchange': 'first',
            'Lot_Size': 'first'
        }).reset_index()
        
        # Sort by symbol
        unique_symbols = unique_symbols.sort_values('Symbol')
        
        # Save to CSV
        missing_file = self.output_dir / f"MISSING_MAPPINGS_{self.timestamp}.csv"
        unique_symbols.to_csv(missing_file, index=False)
        
        # Also create a template for easy addition to mapping file
        template_file = self.output_dir / f"MAPPING_TEMPLATE_{self.timestamp}.csv"
        template_df = unique_symbols[['Symbol', 'Suggested_Ticker', 'Underlying', 'Exchange', 'Lot_Size']]
        template_df.columns = ['Symbol', 'Ticker', 'Underlying', 'Exchange', 'Lot_Size']
        template_df.to_csv(template_file, index=False)
        
        logger.info(f"Created missing mappings report with {len(unique_symbols)} unmapped symbols")
        logger.info(f"Missing mappings report: {missing_file}")
        logger.info(f"Mapping template file: {template_file}")
        
        return missing_file
    
    def _suggest_ticker(self, symbol: str) -> str:
        """
        Suggest a ticker based on common patterns
        """
        symbol_upper = symbol.upper()
        
        # Remove common suffixes
        cleaned = symbol_upper
        for suffix in ['EQ', 'FUT', 'OPT', 'CE', 'PE', '-EQ', '-FUT']:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
                break
        
        # Common index mappings
        index_map = {
            'NIFTY': 'NZ',
            'BANKNIFTY': 'AF1',
            'FINNIFTY': 'FINNIFTY',
            'MIDCPNIFTY': 'MIDCPNIFTY'
        }
        
        for key, value in index_map.items():
            if key in symbol_upper:
                return value
        
        # For others, return cleaned version
        return cleaned.strip('-').strip()
    
    def _create_summary_report(self,
                              parsed_trades_df: pd.DataFrame,
                              starting_positions_df: pd.DataFrame,
                              processed_trades_df: pd.DataFrame,
                              final_positions_df: pd.DataFrame,
                              input_parser=None,
                              trade_parser=None) -> Path:
        """Create a summary report of the processing including missing mappings"""
        summary_file = self.output_dir / f"summary_report_{self.timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRADE PROCESSING SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Missing mappings section
            missing_count = 0
            if input_parser and hasattr(input_parser, 'unmapped_symbols'):
                missing_count += len(input_parser.unmapped_symbols)
            if trade_parser and hasattr(trade_parser, 'unmapped_symbols'):
                missing_count += len(trade_parser.unmapped_symbols)
            
            if missing_count > 0:
                f.write("⚠️  MISSING MAPPINGS:\n")
                f.write("-" * 30 + "\n")
                
                if input_parser and hasattr(input_parser, 'unmapped_symbols') and input_parser.unmapped_symbols:
                    f.write(f"Position file: {len(input_parser.unmapped_symbols)} unmapped symbols\n")
                    unique_pos_symbols = set(item['symbol'] for item in input_parser.unmapped_symbols)
                    f.write(f"  Symbols: {', '.join(sorted(unique_pos_symbols)[:10])}")
                    if len(unique_pos_symbols) > 10:
                        f.write(f" ... and {len(unique_pos_symbols) - 10} more")
                    f.write("\n")
                
                if trade_parser and hasattr(trade_parser, 'unmapped_symbols') and trade_parser.unmapped_symbols:
                    f.write(f"Trade file: {len(trade_parser.unmapped_symbols)} unmapped symbols\n")
                    unique_trade_symbols = set(item['symbol'] for item in trade_parser.unmapped_symbols)
                    f.write(f"  Symbols: {', '.join(sorted(unique_trade_symbols)[:10])}")
                    if len(unique_trade_symbols) > 10:
                        f.write(f" ... and {len(unique_trade_symbols) - 10} more")
                    f.write("\n")
                
                f.write("\n📝 Check MISSING_MAPPINGS_*.csv for complete list\n")
                f.write("📝 Use MAPPING_TEMPLATE_*.csv to add to your mapping file\n\n")
            
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
