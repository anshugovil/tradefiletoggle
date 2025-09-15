"""
Trade Processor Module - FIXED QTY CALCULATION VERSION
Correctly assigns strategies and calculates QTY as lots * lot_size to avoid decimals
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
    original_row_index: int
    original_trade: dict
    bloomberg_ticker: str
    strategy: str
    is_split: bool
    is_opposite: bool
    split_lots: float
    split_qty: float
    lot_size: int  # Added to track lot size


class TradeProcessor:
    """Processes trades against positions to assign strategies"""
    
    def __init__(self, position_manager):
        self.position_manager = position_manager
        self.processed_trades = []
        
    def process_trades(self, trades: List, trade_df: pd.DataFrame) -> pd.DataFrame:
        """Process all trades sequentially"""
        processed_rows = []
        
        # Check for headers
        has_headers = self._check_for_headers(trade_df)
        header_row = None
        data_start_idx = 0
        
        if has_headers:
            header_row = trade_df.iloc[0].to_list()
            data_start_idx = 1
            logger.info(f"Detected headers: {header_row}")
        
        logger.info(f"Processing {len(trades)} trades")
        
        # Process each trade
        for trade_idx, trade in enumerate(trades):
            actual_row_idx = trade_idx + data_start_idx
            
            if actual_row_idx >= len(trade_df):
                logger.warning(f"Trade {trade_idx} exceeds dataframe rows")
                continue
            
            original_row = trade_df.iloc[actual_row_idx]
            
            # Log position before processing
            pos = self.position_manager.get_position(trade.bloomberg_ticker)
            if pos:
                logger.info(f"Before trade {trade_idx}: Position={pos.quantity} ({pos.strategy}), Trade={trade.position_lots}")
            else:
                logger.info(f"Before trade {trade_idx}: No position, Trade={trade.position_lots}")
            
            # Process the trade
            processed = self._process_single_trade(trade, original_row, actual_row_idx)
            processed_rows.extend(processed)
            
            # Log results
            for p in processed:
                logger.info(f"  Result: {p.split_lots} lots, Strategy={p.strategy}, Split={p.is_split}, Opposite={p.is_opposite}")
            
            # Log position after processing
            pos_after = self.position_manager.get_position(trade.bloomberg_ticker)
            if pos_after:
                logger.info(f"After trade {trade_idx}: Position={pos_after.quantity} ({pos_after.strategy})")
            else:
                logger.info(f"After trade {trade_idx}: No position (closed)")
        
        return self._create_output_dataframe(processed_rows, trade_df, has_headers, header_row)
    
    def _check_for_headers(self, trade_df: pd.DataFrame) -> bool:
        """Check if first row contains headers"""
        if len(trade_df) == 0:
            return False
        
        first_row = trade_df.iloc[0]
        header_keywords = ['symbol', 'expiry', 'strike', 'option', 'instr', 'qty', 'price', 'lots']
        first_row_str = ' '.join([str(val).lower() for val in first_row if pd.notna(val)])
        
        if any(keyword in first_row_str for keyword in header_keywords):
            return True
        
        try:
            if pd.notna(first_row.iloc[7]):
                float(first_row.iloc[7])
            if pd.notna(first_row.iloc[12]):
                float(first_row.iloc[12])
            return False
        except (ValueError, TypeError):
            return True
    
    def _process_single_trade(self, trade, original_row: pd.Series, row_index: int) -> List[ProcessedTrade]:
        """Process a single trade, potentially splitting it"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots  # Has sign
        security_type = trade.security_type
        lot_size = trade.lot_size
        
        # Get current position
        position = self.position_manager.get_position(ticker)
        
        if position is None:
            # NEW POSITION - no existing position
            strategy = self._get_new_position_strategy(trade_quantity, security_type)
            
            # Calculate QTY as lots * lot_size (always whole number)
            qty = abs(trade_quantity) * lot_size
            
            processed = ProcessedTrade(
                original_row_index=row_index,
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=False,  # New position, so no opposite
                split_lots=abs(trade_quantity),
                split_qty=qty,
                lot_size=lot_size
            )
            
            # Update position with strategy
            self.position_manager.update_position(ticker, trade_quantity, security_type, strategy)
            return [processed]
        
        # Check if trade opposes position
        is_opposing = self.position_manager.is_trade_opposing(ticker, trade_quantity, security_type)
        
        if not is_opposing:
            # SAME DIRECTION - adding to position
            strategy = position.strategy  # Keep position's strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            # Calculate QTY as lots * lot_size
            qty = abs(trade_quantity) * lot_size
            
            processed = ProcessedTrade(
                original_row_index=row_index,
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=qty,
                lot_size=lot_size
            )
            
            # Update position, keeping the same strategy
            self.position_manager.update_position(ticker, trade_quantity, security_type, strategy)
            return [processed]
        
        # OPPOSING TRADE - check if split needed
        if abs(trade_quantity) <= abs(position.quantity):
            # NO SPLIT - trade just reduces position
            strategy = position.strategy  # INHERIT position's strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            # Calculate QTY as lots * lot_size
            qty = abs(trade_quantity) * lot_size
            
            processed = ProcessedTrade(
                original_row_index=row_index,
                original_trade=original_row.to_dict(),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=qty,
                lot_size=lot_size
            )
            
            # Update position, keeping the same strategy (position not flipped)
            self.position_manager.update_position(ticker, trade_quantity, security_type, strategy)
            return [processed]
        
        # SPLIT NEEDED
        logger.info(f"SPLITTING: Position={position.quantity} ({position.strategy}), Trade={trade_quantity}")
        return self._split_trade(trade, original_row, position, row_index)
    
    def _split_trade(self, trade, original_row: pd.Series, position, row_index: int) -> List[ProcessedTrade]:
        """Split a trade that exceeds position size"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots
        security_type = trade.security_type
        lot_size = trade.lot_size
        
        # Calculate split quantities
        close_quantity = -position.quantity  # Opposite sign to close
        remaining_quantity = trade_quantity + position.quantity  # What's left after closing
        
        # Calculate split lots
        close_lots = abs(position.quantity)
        open_lots = abs(remaining_quantity)
        
        # Calculate QTY as lots * lot_size (avoids decimals)
        close_qty = close_lots * lot_size
        open_qty = open_lots * lot_size
        
        # FIRST SPLIT - CLOSING EXISTING POSITION
        # This MUST inherit the position's strategy
        close_strategy = position.strategy  # ALWAYS inherit position's strategy when closing
        is_opposite_close = self._is_strategy_opposite_to_trade(close_strategy, trade_quantity, security_type)
        
        processed_close = ProcessedTrade(
            original_row_index=row_index,
            original_trade=original_row.to_dict(),
            bloomberg_ticker=ticker,
            strategy=close_strategy,  # INHERIT from position being closed
            is_split=True,
            is_opposite=is_opposite_close,
            split_lots=close_lots,
            split_qty=close_qty,
            lot_size=lot_size
        )
        
        logger.info(f"  Split 1 (close): {close_lots} lots ({close_qty} qty), Strategy={close_strategy}")
        
        # Update position to zero (closing)
        self.position_manager.update_position(ticker, close_quantity, security_type, close_strategy)
        
        # SECOND SPLIT - OPENING NEW POSITION
        # This gets new strategy based on trade direction
        new_strategy = self._get_new_position_strategy(remaining_quantity, security_type)
        is_opposite_open = self._is_strategy_opposite_to_trade(new_strategy, remaining_quantity, security_type)
        
        processed_open = ProcessedTrade(
            original_row_index=row_index,
            original_trade=original_row.to_dict(),
            bloomberg_ticker=ticker,
            strategy=new_strategy,  # NEW strategy for new position
            is_split=True,
            is_opposite=is_opposite_open,
            split_lots=open_lots,
            split_qty=open_qty,
            lot_size=lot_size
        )
        
        logger.info(f"  Split 2 (open): {open_lots} lots ({open_qty} qty), Strategy={new_strategy}")
        
        # Update position with new position AND NEW STRATEGY
        self.position_manager.update_position(ticker, remaining_quantity, security_type, new_strategy)
        
        return [processed_close, processed_open]
    
    def _get_new_position_strategy(self, trade_quantity: float, security_type: str) -> str:
        """Get strategy for a NEW position (not closing existing)"""
        if security_type == 'Put':
            # Puts are inverted
            if trade_quantity > 0:
                return 'FUSH'  # Long put = short exposure
            else:
                return 'FULO'  # Short put = long exposure
        else:
            # Futures and Calls
            if trade_quantity > 0:
                return 'FULO'  # Long futures/call = long exposure
            else:
                return 'FUSH'  # Short futures/call = short exposure
    
    def _is_strategy_opposite_to_trade(self, strategy: str, trade_quantity: float, security_type: str) -> bool:
        """
        Check if strategy is opposite to trade direction
        This determines the 'Opposite?' flag
        """
        if security_type == 'Put':
            # For puts: FUSH = long put, FULO = short put
            if strategy == 'FUSH':
                # FUSH strategy means we're long puts (short exposure)
                # So a sell trade (negative) would be opposite
                return trade_quantity < 0
            else:  # FULO
                # FULO strategy means we're short puts (long exposure)
                # So a buy trade (positive) would be opposite
                return trade_quantity > 0
        else:
            # For futures/calls
            if strategy == 'FULO':
                # FULO means we're long
                # So a sell trade (negative) would be opposite
                return trade_quantity < 0
            else:  # FUSH
                # FUSH means we're short
                # So a buy trade (positive) would be opposite
                return trade_quantity > 0
    
    def _create_output_dataframe(self, processed_trades: List[ProcessedTrade], 
                                original_df: pd.DataFrame, has_headers: bool,
                                header_row: Optional[List]) -> pd.DataFrame:
        """Create output DataFrame with all columns and proper headers"""
        output_rows = []
        
        for pt in processed_trades:
            row_dict = deepcopy(pt.original_trade)
            
            # Update quantities for splits (QTY = lots * lot_size, no decimals)
            if pt.is_split:
                buy_sell = str(row_dict.get(10, '')).upper()
                sign = -1 if buy_sell.startswith('S') else 1
                
                # Use calculated QTY (lots * lot_size) instead of proportional split
                row_dict[11] = pt.split_qty * sign  # QTY with sign
                row_dict[12] = pt.split_lots * sign  # Lots with sign
            else:
                # For non-split trades, also ensure QTY is lots * lot_size
                buy_sell = str(row_dict.get(10, '')).upper()
                sign = -1 if buy_sell.startswith('S') else 1
                
                # Recalculate QTY to ensure no decimals
                row_dict[11] = pt.split_qty * sign  # QTY = lots * lot_size
                row_dict[12] = pt.split_lots * sign  # Lots
            
            # Add new columns
            row_dict['Strategy'] = pt.strategy
            row_dict['Split?'] = 'Yes' if pt.is_split else 'No'
            row_dict['Opposite?'] = 'Yes' if pt.is_opposite else 'No'
            row_dict['Bloomberg_Ticker'] = pt.bloomberg_ticker
            
            output_rows.append(row_dict)
        
        result_df = pd.DataFrame(output_rows)
        
        # Set column names
        if has_headers and header_row:
            original_headers = header_row[:14]
            new_headers = ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
            all_headers = original_headers + new_headers
            
            column_mapping = {}
            for i in range(14):
                column_mapping[i] = original_headers[i] if i < len(original_headers) else f'Col_{i}'
            
            result_df.rename(columns=column_mapping, inplace=True)
            final_columns = original_headers + new_headers
        else:
            final_columns = list(range(14)) + ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
        
        # Ensure all columns exist and are in correct order
        for col in final_columns:
            if col not in result_df.columns:
                result_df[col] = None
        
        result_df = result_df[final_columns]
        
        # Log to verify no decimals in QTY
        if has_headers and 'Qty' in result_df.columns:
            qty_col = 'Qty'
        elif 11 in result_df.columns:
            qty_col = 11
        else:
            qty_col = None
        
        if qty_col and qty_col in result_df.columns:
            qty_values = result_df[qty_col].dropna()
            if len(qty_values) > 0:
                has_decimals = any(val != int(val) for val in qty_values if pd.notna(val) and val != 0)
                if has_decimals:
                    logger.warning("Warning: Some QTY values still have decimals")
                else:
                    logger.info("✓ All QTY values are whole numbers")
        
        return result_df
