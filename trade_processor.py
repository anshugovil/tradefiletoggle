"""
Trade Processor Module - FIXED VERSION
Properly maintains row mapping and preserves headers
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class ProcessedTrade:
    """Represents a processed trade with strategy assignment"""
    original_row_index: int  # Track original row index
    original_trade: dict
    bloomberg_ticker: str
    strategy: str
    is_split: bool
    is_opposite: bool
    split_lots: float
    split_qty: float


class TradeProcessor:
    """Processes trades against positions to assign strategies"""
    
    def __init__(self, position_manager):
        self.position_manager = position_manager
        self.processed_trades = []
        
    def process_trades(self, trades: List, trade_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all trades sequentially
        trades: List of Position objects from trade parser
        trade_df: Original trade DataFrame for preserving structure
        """
        processed_rows = []
        
        # Check if first row is headers
        has_headers = self._check_for_headers(trade_df)
        header_row = None
        data_start_idx = 0
        
        if has_headers:
            header_row = trade_df.iloc[0].to_list()
            data_start_idx = 1
            logger.info(f"Detected headers in trade file: {header_row}")
        
        logger.info(f"Processing {len(trades)} trades starting from row {data_start_idx}")
        
        # Process each trade with its corresponding row
        for trade_idx, trade in enumerate(trades):
            # The actual row index in the original dataframe
            actual_row_idx = trade_idx + data_start_idx
            
            if actual_row_idx >= len(trade_df):
                logger.warning(f"Trade {trade_idx} exceeds dataframe rows")
                continue
            
            # Get the exact corresponding row
            original_row = trade_df.iloc[actual_row_idx]
            
            # Verify this is the right row by checking symbol
            row_symbol = str(original_row.iloc[5]).strip().upper() if pd.notna(original_row.iloc[5]) else ""
            trade_symbol = trade.symbol.upper()
            
            if row_symbol != trade_symbol:
                logger.warning(f"Row mismatch at index {actual_row_idx}: row symbol={row_symbol}, trade symbol={trade_symbol}")
                # Try to find the correct row
                original_row = self._find_correct_row(trade, trade_df, data_start_idx)
                if original_row is None:
                    logger.error(f"Could not find matching row for trade {trade_symbol}")
                    continue
            
            # Process this trade with its original row
            processed = self._process_single_trade(trade, original_row, actual_row_idx)
            processed_rows.extend(processed)
        
        # Create output dataframe with headers if present
        return self._create_output_dataframe(processed_rows, trade_df, has_headers, header_row)
    
    def _check_for_headers(self, trade_df: pd.DataFrame) -> bool:
        """Check if the first row contains headers"""
        if len(trade_df) == 0:
            return False
        
        first_row = trade_df.iloc[0]
        
        # Check if first row contains typical header keywords
        header_keywords = ['symbol', 'expiry', 'strike', 'option', 'instr', 'qty', 'price', 'lots', 'code', 'scheme']
        first_row_str = ' '.join([str(val).lower() for val in first_row if pd.notna(val)])
        
        if any(keyword in first_row_str for keyword in header_keywords):
            return True
        
        # Check if numeric columns have non-numeric values in first row
        try:
            # Column 7 (Lot Size) and 12 (Lots Traded) should be numeric
            if pd.notna(first_row.iloc[7]):
                float(first_row.iloc[7])
            if pd.notna(first_row.iloc[12]):
                float(first_row.iloc[12])
            return False  # Successfully converted to numbers, so not headers
        except (ValueError, TypeError):
            return True  # Failed to convert, likely headers
    
    def _find_correct_row(self, trade, trade_df: pd.DataFrame, start_idx: int) -> Optional[pd.Series]:
        """Find the correct row for a trade"""
        trade_symbol = trade.symbol.upper()
        trade_lots = abs(trade.position_lots)
        
        # Search for matching row
        for idx in range(start_idx, len(trade_df)):
            row = trade_df.iloc[idx]
            row_symbol = str(row.iloc[5]).strip().upper() if pd.notna(row.iloc[5]) else ""
            
            if row_symbol == trade_symbol:
                try:
                    row_lots = abs(float(row.iloc[12])) if pd.notna(row.iloc[12]) else 0
                    if abs(row_lots - trade_lots) < 0.001:
                        logger.info(f"Found matching row at index {idx} for {trade_symbol}")
                        return row
                except:
                    continue
        
        return None
    
    def _process_single_trade(self, trade, original_row: pd.Series, row_index: int) -> List[ProcessedTrade]:
        """Process a single trade, potentially splitting it"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots  # Already has sign
        security_type = trade.security_type
        
        logger.debug(f"Processing row {row_index}: {trade.symbol}, qty={trade_quantity}, ticker={ticker}")
        
        # Get current position
        position = self.position_manager.get_position(ticker)
        
        if position is None:
            # No existing position - new position
            strategy = self._get_new_position_strategy(trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_row_index=row_index,
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=False,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            
            return [processed]
        
        # Check if trade opposes position
        is_opposing = self.position_manager.is_trade_opposing(ticker, trade_quantity, security_type)
        
        if not is_opposing:
            # Same direction - add to position
            strategy = position.strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_row_index=row_index,
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            
            return [processed]
        
        # Opposing trade - check if split needed
        if abs(trade_quantity) <= abs(position.quantity):
            # No split needed
            strategy = position.strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_row_index=row_index,
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            
            return [processed]
        
        # Split needed
        logger.info(f"Splitting trade at row {row_index}: position={position.quantity}, trade={trade_quantity}")
        return self._split_trade(trade, original_row, position, row_index)
    
    def _split_trade(self, trade, original_row: pd.Series, position, row_index: int) -> List[ProcessedTrade]:
        """Split a trade that exceeds position size"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots
        security_type = trade.security_type
        
        # Calculate split
        close_quantity = -position.quantity
        remaining_quantity = trade_quantity + position.quantity
        
        # Calculate split ratios
        total_lots = abs(trade_quantity)
        close_lots = abs(position.quantity)
        open_lots = abs(remaining_quantity)
        
        # Split QTY proportionally
        total_qty = abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
        close_qty = total_qty * (close_lots / total_lots)
        open_qty = total_qty * (open_lots / total_lots)
        
        # First split - closing position
        is_opposite_close = self._is_strategy_opposite_to_trade(position.strategy, trade_quantity, security_type)
        processed_close = ProcessedTrade(
            original_row_index=row_index,
            original_trade=original_row.to_dict(),
            bloomberg_ticker=ticker,
            strategy=position.strategy,
            is_split=True,
            is_opposite=is_opposite_close,
            split_lots=close_lots,
            split_qty=close_qty
        )
        
        # Update position (should be zero)
        self.position_manager.update_position(ticker, close_quantity, security_type)
        
        # Second split - opening new position
        new_strategy = self._get_new_position_strategy(remaining_quantity, security_type)
        is_opposite_open = self._is_strategy_opposite_to_trade(new_strategy, remaining_quantity, security_type)
        
        processed_open = ProcessedTrade(
            original_row_index=row_index,
            original_trade=original_row.to_dict(),
            bloomberg_ticker=ticker,
            strategy=new_strategy,
            is_split=True,
            is_opposite=is_opposite_open,
            split_lots=open_lots,
            split_qty=open_qty
        )
        
        # Update position with new position
        self.position_manager.update_position(ticker, remaining_quantity, security_type)
        
        return [processed_close, processed_open]
    
    def _get_new_position_strategy(self, trade_quantity: float, security_type: str) -> str:
        """Get strategy for a new position"""
        if security_type == 'Put':
            # Puts are inverted
            return 'FUSH' if trade_quantity > 0 else 'FULO'
        else:
            # Futures and Calls
            return 'FULO' if trade_quantity > 0 else 'FUSH'
    
    def _is_strategy_opposite_to_trade(self, strategy: str, trade_quantity: float, security_type: str) -> bool:
        """Check if strategy is opposite to trade direction"""
        if security_type == 'Put':
            if strategy == 'FUSH':
                return trade_quantity < 0  # Selling when should be buying
            else:  # FULO
                return trade_quantity > 0  # Buying when should be selling
        else:
            if strategy == 'FULO':
                return trade_quantity < 0  # Selling when should be buying
            else:  # FUSH
                return trade_quantity > 0  # Buying when should be selling
    
    def _create_output_dataframe(self, processed_trades: List[ProcessedTrade], 
                                original_df: pd.DataFrame, has_headers: bool,
                                header_row: Optional[List]) -> pd.DataFrame:
        """Create output DataFrame with all columns and proper headers"""
        output_rows = []
        
        # Process each trade
        for pt in processed_trades:
            row_dict = deepcopy(pt.original_trade)
            
            # Update quantities for splits
            if pt.is_split:
                # Determine sign from B/S column
                buy_sell = str(row_dict.get(10, '')).upper()
                sign = -1 if buy_sell.startswith('S') else 1
                
                row_dict[11] = pt.split_qty * sign  # Qty
                row_dict[12] = pt.split_lots * sign  # Lots Traded
            
            # Add new columns
            row_dict['Strategy'] = pt.strategy
            row_dict['Split?'] = 'Yes' if pt.is_split else 'No'
            row_dict['Opposite?'] = 'Yes' if pt.is_opposite else 'No'
            row_dict['Bloomberg_Ticker'] = pt.bloomberg_ticker
            
            output_rows.append(row_dict)
        
        # Create DataFrame
        result_df = pd.DataFrame(output_rows)
        
        # Set column names
        if has_headers and header_row:
            # Use original headers plus new columns
            original_headers = header_row[:14]  # First 14 columns
            new_headers = ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
            all_headers = original_headers + new_headers
            
            # Rename columns to match headers
            column_mapping = {}
            for i in range(14):
                column_mapping[i] = original_headers[i] if i < len(original_headers) else f'Col_{i}'
            
            result_df.rename(columns=column_mapping, inplace=True)
            
            # Ensure all columns are present in correct order
            final_columns = original_headers + new_headers
        else:
            # No headers - use numeric columns
            final_columns = list(range(14)) + ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
        
        # Ensure all columns exist and are in correct order
        for col in final_columns:
            if col not in result_df.columns:
                result_df[col] = None
        
        # Reorder columns
        result_df = result_df[final_columns]
        
        logger.info(f"Created output dataframe with {len(result_df)} rows and columns: {list(result_df.columns)}")
        
        return result_df
