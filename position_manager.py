"""
Position Manager Module - FIXED VERSION
Correctly maintains strategy when positions flip
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PositionState:
    """Represents current position state for a ticker"""
    ticker: str
    quantity: float  # in lots (positive for long, negative for short)
    strategy: str  # FULO or FUSH
    security_type: str  # Futures, Call, Put
    
    def update_position(self, trade_quantity: float) -> 'PositionState':
        """
        Update position after a trade
        IMPORTANT: Strategy only changes when position flips sign
        """
        new_quantity = self.quantity + trade_quantity
        
        # Determine new strategy ONLY if position flips sign
        if (self.quantity > 0 and new_quantity < 0) or (self.quantity < 0 and new_quantity > 0):
            # Position flipped - determine new strategy based on NEW position
            if new_quantity > 0:
                # Now long
                if self.security_type == 'Put':
                    new_strategy = 'FUSH'  # Long put = short exposure
                else:
                    new_strategy = 'FULO'  # Long futures/call = long exposure
            else:
                # Now short
                if self.security_type == 'Put':
                    new_strategy = 'FULO'  # Short put = long exposure
                else:
                    new_strategy = 'FUSH'  # Short futures/call = short exposure
        else:
            # No flip, keep same strategy
            new_strategy = self.strategy
            
        return PositionState(
            ticker=self.ticker,
            quantity=new_quantity,
            strategy=new_strategy,
            security_type=self.security_type
        )


class PositionManager:
    """Manages positions throughout trade processing"""
    
    def __init__(self):
        self.positions: Dict[str, PositionState] = {}
        self.initial_positions: Dict[str, PositionState] = {}
        
    def initialize_from_positions(self, positions) -> pd.DataFrame:
        """
        Initialize position states from parsed position file
        Returns DataFrame of starting positions
        """
        starting_positions = []
        
        for position in positions:
            ticker_key = position.bloomberg_ticker
            
            # Determine initial strategy based on position direction and type
            if position.security_type == 'Put':
                # Puts are inverted: Long Put = FUSH, Short Put = FULO
                if position.position_lots > 0:
                    strategy = 'FUSH'  # Long put = short exposure
                else:
                    strategy = 'FULO'  # Short put = long exposure
            else:
                # Futures and Calls: Long = FULO, Short = FUSH
                if position.position_lots > 0:
                    strategy = 'FULO'  # Long futures/call = long exposure
                else:
                    strategy = 'FUSH'  # Short futures/call = short exposure
            
            pos_state = PositionState(
                ticker=ticker_key,
                quantity=position.position_lots,  # Keep the sign
                strategy=strategy,
                security_type=position.security_type
            )
            
            self.positions[ticker_key] = pos_state
            self.initial_positions[ticker_key] = pos_state
            
            starting_positions.append({
                'Ticker': ticker_key,
                'QTY': position.position_lots
            })
        
        return pd.DataFrame(starting_positions)
    
    def get_position(self, ticker: str) -> Optional[PositionState]:
        """Get current position for a ticker"""
        return self.positions.get(ticker)
    
    def update_position(self, ticker: str, trade_quantity: float, security_type: str):
        """
        Update position after processing a trade
        IMPORTANT: This should be called AFTER the trade has been processed
        """
        if ticker in self.positions:
            # Update existing position
            old_pos = self.positions[ticker]
            new_pos = old_pos.update_position(trade_quantity)
            self.positions[ticker] = new_pos
            
            logger.debug(f"Position update for {ticker}:")
            logger.debug(f"  Old: {old_pos.quantity} ({old_pos.strategy})")
            logger.debug(f"  Trade: {trade_quantity}")
            logger.debug(f"  New: {new_pos.quantity} ({new_pos.strategy})")
        else:
            # Create new position
            if trade_quantity > 0:
                # Long position
                if security_type == 'Put':
                    strategy = 'FUSH'  # Long put = short exposure
                else:
                    strategy = 'FULO'  # Long futures/call = long exposure
            else:
                # Short position
                if security_type == 'Put':
                    strategy = 'FULO'  # Short put = long exposure
                else:
                    strategy = 'FUSH'  # Short futures/call = short exposure
            
            self.positions[ticker] = PositionState(
                ticker=ticker,
                quantity=trade_quantity,
                strategy=strategy,
                security_type=security_type
            )
            
            logger.debug(f"New position for {ticker}: {trade_quantity} ({strategy})")
    
    def get_final_positions(self) -> pd.DataFrame:
        """Get final positions as DataFrame"""
        final_positions = []
        for ticker, pos_state in self.positions.items():
            if abs(pos_state.quantity) > 0.0001:  # Skip near-zero positions
                final_positions.append({
                    'Ticker': ticker,
                    'QTY': pos_state.quantity
                })
        
        return pd.DataFrame(final_positions)
    
    def is_trade_opposing(self, ticker: str, trade_quantity: float, security_type: str) -> bool:
        """
        Check if trade opposes current position
        A trade opposes if it reduces the absolute position size
        """
        pos = self.get_position(ticker)
        if not pos:
            return False
        
        # Trade opposes if signs are different
        # Long position (positive) vs Sell trade (negative)
        # Short position (negative) vs Buy trade (positive)
        return (pos.quantity > 0 and trade_quantity < 0) or (pos.quantity < 0 and trade_quantity > 0)
