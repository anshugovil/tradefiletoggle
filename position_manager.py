"""
Position Manager Module
Manages position tracking and updates throughout trade processing
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
    quantity: float  # in lots
    strategy: str  # FULO or FUSH
    security_type: str  # Futures, Call, Put
    
    def get_directional_quantity(self) -> float:
        """Get directional quantity accounting for put inversions"""
        if self.security_type == 'Put':
            # For puts: positive quantity with FUSH = short exposure
            #          negative quantity with FULO = long exposure
            return self.quantity
        else:
            # For futures/calls: straightforward
            return self.quantity
    
    def update_position(self, trade_quantity: float) -> 'PositionState':
        """Update position after a trade"""
        new_quantity = self.quantity + trade_quantity
        
        # Determine new strategy if position flips
        if self.quantity > 0 and new_quantity < 0:
            # Flipped from long to short
            if self.security_type == 'Put':
                new_strategy = 'FULO'  # Short put = long exposure
            else:
                new_strategy = 'FUSH'  # Short futures/call = short exposure
        elif self.quantity < 0 and new_quantity > 0:
            # Flipped from short to long
            if self.security_type == 'Put':
                new_strategy = 'FUSH'  # Long put = short exposure
            else:
                new_strategy = 'FULO'  # Long futures/call = long exposure
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
                quantity=position.position_lots,
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
        """Update position after processing a trade"""
        if ticker in self.positions:
            # Update existing position
            old_pos = self.positions[ticker]
            new_pos = old_pos.update_position(trade_quantity)
            self.positions[ticker] = new_pos
        else:
            # Create new position
            if security_type == 'Put':
                strategy = 'FUSH' if trade_quantity > 0 else 'FULO'
            else:
                strategy = 'FULO' if trade_quantity > 0 else 'FUSH'
            
            self.positions[ticker] = PositionState(
                ticker=ticker,
                quantity=trade_quantity,
                strategy=strategy,
                security_type=security_type
            )
    
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
        """Check if trade opposes current position"""
        pos = self.get_position(ticker)
        if not pos:
            return False
        
        # For futures/calls: buy (+) vs short position (-) or sell (-) vs long position (+)
        # For puts: logic is same at position level
        if pos.quantity > 0:
            return trade_quantity < 0  # Long position, sell trade
        else:
            return trade_quantity > 0  # Short position, buy trade
