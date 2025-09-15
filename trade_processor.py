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
        
        # Log trade processing
        logger.info(f"Processing {len(trades)} trades")
        
        # Create a mapping between parsed trades and original rows
        # We'll match by row index primarily
        for idx, trade in enumerate(trades):
            try:
                # Use row index directly if available
                if idx < len(trade_df):
                    original_row = trade_df.iloc[idx]
                else:
                    # Fallback to finding matching row
                    original_row = self._find_matching_row(trade, trade_df, idx)
                
                if original_row is None:
                    logger.warning(f"Could not find matching row for trade {idx}: {trade.symbol}")
                    # Create a synthetic row with the trade data
                    original_row = self._create_synthetic_row(trade)
                
                # Process this trade
                processed = self._process_single_trade(trade, original_row)
                processed_rows.extend(processed)
                
                logger.debug(f"Processed trade {idx}: {trade.symbol} - {len(processed)} output rows")
                
            except Exception as e:
                logger.error(f"Error processing trade {idx}: {e}")
                continue
        
        if not processed_rows:
            logger.warning("No trades were successfully processed")
            # Return empty dataframe with correct structure
            return self._create_empty_output_dataframe(trade_df)
        
        return self._create_output_dataframe(processed_rows, trade_df)
    
    def _create_synthetic_row(self, trade) -> pd.Series:
        """Create a synthetic row when no matching row is found"""
        # Create a row with the basic trade information
        row_data = {
            0: '',  # CP Code
            1: 0,   # TM Code
            2: '',  # Scheme
            3: '',  # TM Name
            4: '',  # Instr
            5: trade.symbol,  # Symbol
            6: trade.expiry_date.strftime('%d-%b-%Y') if hasattr(trade.expiry_date, 'strftime') else str(trade.expiry_date),  # Expiry
            7: trade.lot_size,  # Lot Size
            8: trade.strike_price,  # Strike Price
            9: 'CE' if trade.security_type == 'Call' else 'PE' if trade.security_type == 'Put' else '',  # Option Type
            10: 'B' if trade.position_lots > 0 else 'S',  # B/S
            11: abs(trade.position_lots * trade.lot_size),  # Qty
            12: abs(trade.position_lots),  # Lots Traded
            13: 0  # Avg Price
        }
        return pd.Series(row_data)
    
    def _find_matching_row(self, trade, trade_df: pd.DataFrame, idx: int) -> Optional[pd.Series]:
        """Find the original row in trade_df that matches this trade"""
        # Try direct index first
        if idx < len(trade_df):
            return trade_df.iloc[idx]
        
        # Fallback: match based on symbol and lots
        for row_idx, row in trade_df.iterrows():
            try:
                # Check if symbol matches (column 5)
                row_symbol = str(row.iloc[5]).strip().upper() if pd.notna(row.iloc[5]) else ""
                trade_symbol = trade.symbol.upper()
                
                # Check if lots match (column 12)
                row_lots = abs(float(row.iloc[12])) if pd.notna(row.iloc[12]) else 0
                trade_lots = abs(trade.position_lots)
                
                if row_symbol == trade_symbol and abs(row_lots - trade_lots) < 0.001:
                    return row
            except:
                continue
        
        return None
    
    def _process_single_trade(self, trade, original_row: pd.Series) -> List[ProcessedTrade]:
        """Process a single trade, potentially splitting it"""
        ticker = trade.bloomberg_ticker
        trade_quantity = trade.position_lots  # Already has sign (+ for buy, - for sell)
        security_type = trade.security_type
        
        logger.debug(f"Processing {ticker}: quantity={trade_quantity}, type={security_type}")
        
        # Get current position
        position = self.position_manager.get_position(ticker)
        
        if position is None:
            # No existing position - new position
            strategy = self._get_new_position_strategy(trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_trade=original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=False,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            logger.debug(f"New position created: {strategy}")
            
            return [processed]
        
        # Check if trade opposes position
        is_opposing = self.position_manager.is_trade_opposing(ticker, trade_quantity, security_type)
        
        if not is_opposing:
            # Same direction - add to position
            strategy = position.strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_trade=original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            logger.debug(f"Adding to position: {strategy}, opposite={is_opposite}")
            
            return [processed]
        
        # Opposing trade - check if split needed
        if abs(trade_quantity) <= abs(position.quantity):
            # No split needed - trade closes/reduces position
            strategy = position.strategy
            is_opposite = self._is_strategy_opposite_to_trade(strategy, trade_quantity, security_type)
            
            processed = ProcessedTrade(
                original_trade=original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row),
                bloomberg_ticker=ticker,
                strategy=strategy,
                is_split=False,
                is_opposite=is_opposite,
                split_lots=abs(trade_quantity),
                split_qty=abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
            )
            
            # Update position
            self.position_manager.update_position(ticker, trade_quantity, security_type)
            logger.debug(f"Reducing position: {strategy}, opposite={is_opposite}")
            
            return [processed]
        
        # Split needed
        logger.debug(f"Splitting trade: position={position.quantity}, trade={trade_quantity}")
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
        total_qty = abs(float(original_row.iloc[11]) if pd.notna(original_row.iloc[11]) else trade_quantity * trade.lot_size)
        close_qty = total_qty * (close_lots / total_lots)
        open_qty = total_qty * (open_lots / total_lots)
        
        # First split - closing position
        is_opposite_close = self._is_strategy_opposite_to_trade(position.strategy, trade_quantity, security_type)
        processed_close = ProcessedTrade(
            original_trade=original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row),
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
            original_trade=original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row),
            bloomberg_ticker=ticker,
            strategy=new_strategy,
            is_split=True,
            is_opposite=is_opposite_open,
            split_lots=open_lots,
            split_qty=open_qty
        )
        
        # Update position with new position
        self.position_manager.update_position(ticker, open_quantity, security_type)
        
        logger.debug(f"Split complete: close={position.strategy}, open={new_strategy}")
        
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
    
    def _create_empty_output_dataframe(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """Create an empty output dataframe with the correct structure"""
        if len(original_df.columns) > 0:
            columns = list(original_df.columns) + ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
        else:
            columns = list(range(14)) + ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
        
        return pd.DataFrame(columns=columns)
    
    def _create_output_dataframe(self, processed_trades: List[ProcessedTrade], original_df: pd.DataFrame) -> pd.DataFrame:
        """Create output DataFrame with all columns"""
        output_rows = []
        
        for pt in processed_trades:
            row_dict = deepcopy(pt.original_trade)
            
            # Update the quantities for splits - maintain sign for B/S column
            if pt.is_split:
                # Determine the original sign from B/S column or trade quantity
                if 10 in row_dict:
                    original_sign = -1 if str(row_dict[10]).upper() == 'S' else 1
                elif isinstance(row_dict, dict) and 'B/S' in row_dict:
                    original_sign = -1 if str(row_dict['B/S']).upper() == 'S' else 1
                else:
                    # Default to positive
                    original_sign = 1
                
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
        if len(original_df.columns) > 0:
            original_cols = list(original_df.columns)
        else:
            original_cols = list(range(14))
        
        new_cols = ['Strategy', 'Split?', 'Opposite?', 'Bloomberg_Ticker']
        all_cols = original_cols + new_cols
        
        # Ensure all columns exist
        for col in all_cols:
            if col not in result_df.columns:
                result_df[col] = None
        
        return result_df[all_cols]
