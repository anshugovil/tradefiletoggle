"""
Trade Processor Module
Core logic for processing trades against positions
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
        
        # Process each trade sequentially
        for idx, trade in enumerate(trades):
            # Find matching row in original dataframe
            original_row = self._find_matching_row(trade, trade_df, idx)
            if original_row is None:
                continue
                
            # Process this trade
            processed = self._process_single_trade(trade, original_row)
            processed_rows.extend(processed)
        
        return self._create_output_dataframe(processed_rows, trade_df)
    
    def _find_matching_row(self, trade, trade_df: pd.DataFrame, idx: int) -> Optional[pd.Series]:
        """Find the original row in trade_df that matches this trade"""
        # For MS format, match based on row index if possible
        if idx < len(trade_df):
            return trade_df.iloc[idx]
        
        # Fallback: match based on symbol and lots
        for idx, row in trade_df.iterrows():
            if (str(row.iloc[5]).upper() == trade.symbol.upper() and
                abs(float(row.iloc[12]) if pd.notna(row.iloc[12]) else 0) == abs(trade.position_lots)):
                return row
        return None
    
    def _process_single_trade(self, trade, original_row: pd.Series) -> List[ProcessedTrade]:
        """Process a single trade, potentially splitting it"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots  # Already has sign (+ for buy, - for sell)
        security_type = trade.security_type
        
        # Get current position
        position = self.position_manager.get_position(ticker)
        
        if position is None:
            # No existing position - new position
            strategy = self._get_new_position_strategy(trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=False,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else 0)
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
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else 0)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            
            return [processed]
        
        # Opposing trade - check if split needed
        if abs(trade_quantity) <= abs(position.quantity):
            # No split needed - trade closes/reduces position
            strategy = position.strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else 0)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            
            return [processed]
        
        # Split needed
        return self._split_trade(trade, original_row, position)
    
    def _split_trade(self, trade, original_row: pd.Series, position) -> List[ProcessedTrade]:
        """Split a trade that exceeds position size"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots
        security_type = trade.security_type
        
        # First split: close existing position
        close_quantity = -position.quantity  # Opposite sign to close
        
        # Calculate split ratios
        total_lots = abs(trade_quantity)
        close_lots = abs(position.quantity)
        open_lots = total_lots - close_lots
        
        # Split the QTY proportionally
        total_qty = abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else 0)
        close_qty = total_qty * (close_lots / total_lots)
        open_qty = total_qty * (open_lots / total_lots)
        
        # First split - closing position
        is_opposite_close = self._is_strategy_opposite_to_trade(position.strategy, trade_quantity, security_type)
        processed_close = ProcessedTrade(
            original_trade=original_row.to_dict(),
            bloomberg_ticker=ticker,
            strategy=position.strategy,
            is_split=True,
            is_opposite=is_opposite_close,
            split_lots=close_lots,
            split_qty=close_qty
        )
        
        # Update position (should be zero after close)
        self.position_manager.update_position(ticker, close_quantity, security_type)
        
        # Second split - opening new position
        open_quantity = trade_quantity + position.quantity  # Remaining after close
        new_strategy = self._get_new_position_strategy(open_quantity, security_type)
        is_opposite_open = self._is_strategy_opposite_to_trade(new_strategy, open_quantity, security_type)
        
        processed_open = ProcessedTrade(
            original_trade=original_row.to_dict(),
            bloomberg_ticker=ticker,
            strategy=new_strategy,
            is_split=True,
            is_opposite=is_opposite_open,
            split_lots=open_lots,
            split_qty=open_qty
        )
        
        # Update position with new position
        self.position_manager.update_position(ticker, open_quantity, security_type)
        
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
            # For puts: FUSH = long put (short exposure), FULO = short put (long exposure)
            if strategy == 'FUSH':
                # FUSH strategy: should be buying puts
                return trade_quantity < 0  # Selling when should be buying
            else:  # FULO
                # FULO strategy: should be selling puts
                return trade_quantity > 0  # Buying when should be selling
        else:
            # For futures/calls: FULO = long, FUSH = short
            if strategy == 'FULO':
                # FULO strategy: should be buying
                return trade_quantity < 0  # Selling when should be buying
            else:  # FUSH
                # FUSH strategy: should be selling
                return trade_quantity > 0  # Buying when should be selling
    
    def _create_output_dataframe(self, processed_trades: List[ProcessedTrade], original_df: pd.DataFrame) -> pd.DataFrame:
        """Create output DataFrame with all columns"""
        output_rows = []
        
        for pt in processed_trades:
            row_dict = deepcopy(pt.original_trade)
            
            # Update the quantities for splits - maintain sign for B/S column
            if pt.is_split:
                # Preserve the sign in column 12 (Lots Traded)
                if isinstance(row_dict, dict):
                    # Dictionary access
                    original_sign = -1 if (12 in row_dict and row_dict[12] < 0) or (row_dict.get(10, '') == 'S') else 1
                    row_dict[12] = pt.split_lots * original_sign
                    row_dict[11] = pt.split_qty * original_sign
                else:
                    # If it's a series/array access
                    original_sign = -1 if float(row_dict.get(12, 0)) < 0 else 1
                    row_dict[12] = pt.split_lots * original_sign
                    row_dict[11] = pt.split_qty * original_sign
            
            # Add new columns
            row_dict['Strategy'] = pt.strategy
            row_dict['Split?'] = 'Yes' if pt.is_split else 'No'
            row_dict['Opposite?'] = 'Yes' if pt.is_opposite else 'No'
            row_dict['Bloomberg_Ticker'] = pt.bloomberg_ticker
            
            output_rows.append(row_dict)
        
        # Create DataFrame
        result_df = pd.DataFrame(output_rows)
        
        # Ensure correct column order
        original_cols = list(original_df.columns) if len(original_df.columns) > 0 else [i for i in range(14)]
        new_cols = ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
        
        # Order columns properly
        all_cols = list(original_cols) + new_cols
        
        # Ensure all columns exist
        for col in all_cols:
            if col not in result_df.columns:
                result_df[col] = None
        
        return result_df[all_cols]
