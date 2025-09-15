"""
Position Manager Module - COMPLETE FIXED VERSION
Properly tracks positions and their strategies, updating strategy when positions flip
Includes all methods required by Streamlit app
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a position with quantity and strategy"""
    ticker: str
    quantity: float  # Signed quantity (positive = long, negative = short)
    security_type: str
    strategy: str  # FULO or FUSH
    
    def __repr__(self):
        return f"Position({self.ticker}, qty={self.quantity}, strategy={self.strategy}, type={self.security_type})"


class PositionManager:
    """Manages positions and their strategies"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.initial_positions_df = None  # Store initial positions for reference
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get current position for a ticker"""
        return self.positions.get(ticker)
    
    def update_position(self, ticker: str, quantity_change: float, 
                       security_type: str, strategy: str):
        """
        Update position with a trade
        
        Args:
            ticker: Bloomberg ticker
            quantity_change: Signed quantity to add (positive = buy, negative = sell)
            security_type: Type of security (Future, Call, Put)
            strategy: Strategy to assign (FULO or FUSH)
        """
        if ticker not in self.positions:
            # New position
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity_change,
                security_type=security_type,
                strategy=strategy
            )
            logger.info(f"Created new position: {self.positions[ticker]}")
        else:
            # Update existing position
            old_position = self.positions[ticker]
            old_quantity = old_position.quantity
            old_strategy = old_position.strategy
            new_quantity = old_quantity + quantity_change
            
            if abs(new_quantity) < 0.0001:  # Effectively zero
                # Position closed
                del self.positions[ticker]
                logger.info(f"Closed position for {ticker} (was {old_quantity} with strategy {old_strategy})")
            else:
                # Update quantity and strategy
                # Strategy changes when position flips or when explicitly set
                self.positions[ticker].quantity = new_quantity
                self.positions[ticker].strategy = strategy
                
                # Log strategy change if it occurred
                if old_strategy != strategy:
                    logger.info(f"Updated {ticker}: qty {old_quantity}->{new_quantity}, "
                              f"strategy {old_strategy}->{strategy} (STRATEGY CHANGED)")
                else:
                    logger.info(f"Updated {ticker}: qty {old_quantity}->{new_quantity}, "
                              f"strategy remains {strategy}")
    
    def initialize_from_positions(self, initial_positions: List) -> pd.DataFrame:
        """
        Initialize position manager with existing positions from input parser
        
        Args:
            initial_positions: List of Position objects from input parser
            
        Returns:
            DataFrame of starting positions
        """
        self.positions.clear()
        positions_data = []
        
        for pos in initial_positions:
            # Determine initial strategy based on position direction and security type
            if pos.security_type == 'Put':
                # Puts are inverted
                strategy = 'FUSH' if pos.position_lots > 0 else 'FULO'
            else:  # Futures or Calls
                strategy = 'FULO' if pos.position_lots > 0 else 'FUSH'
            
            # Create Position object
            self.positions[pos.bloomberg_ticker] = Position(
                ticker=pos.bloomberg_ticker,
                quantity=pos.position_lots,
                security_type=pos.security_type,
                strategy=strategy
            )
            
            # Add to DataFrame data
            positions_data.append({
                'Ticker': pos.bloomberg_ticker,
                'Symbol': pos.symbol,
                'Security_Type': pos.security_type,
                'Expiry': pos.expiry_date,
                'Strike': pos.strike_price if pos.security_type != 'Futures' else 0,
                'QTY': pos.position_lots * pos.lot_size,
                'Lots': pos.position_lots,
                'Lot_Size': pos.lot_size,
                'Strategy': strategy,
                'Direction': 'Long' if pos.position_lots > 0 else 'Short'
            })
            
            logger.info(f"Initialized position: {pos.bloomberg_ticker} with {pos.position_lots} lots, strategy={strategy}")
        
        # Create and store initial positions DataFrame
        self.initial_positions_df = pd.DataFrame(positions_data)
        return self.initial_positions_df
    
    def get_final_positions(self) -> pd.DataFrame:
        """
        Get final positions as a DataFrame
        
        Returns:
            DataFrame of current positions
        """
        positions_data = []
        
        for ticker, position in self.positions.items():
            # Extract details from bloomberg ticker
            symbol = ticker.split(' ')[0].replace('=', '')
            
            # Determine security type from ticker
            if 'C' in ticker and ('/' in ticker):
                security_type = 'Call'
            elif 'P' in ticker and ('/' in ticker):
                security_type = 'Put'
            else:
                security_type = 'Futures'
            
            positions_data.append({
                'Ticker': ticker,
                'Symbol': symbol,
                'Security_Type': position.security_type,
                'QTY': position.quantity * 100,  # Assuming default lot size of 100
                'Lots': position.quantity,
                'Strategy': position.strategy,
                'Direction': 'Long' if position.quantity > 0 else 'Short'
            })
        
        if positions_data:
            return pd.DataFrame(positions_data)
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['Ticker', 'Symbol', 'Security_Type', 'QTY', 'Lots', 'Strategy', 'Direction'])
    
    def is_trade_opposing(self, ticker: str, trade_quantity: float, 
                         security_type: str) -> bool:
        """
        Check if a trade opposes the current position
        
        Args:
            ticker: Bloomberg ticker
            trade_quantity: Signed trade quantity
            security_type: Type of security
            
        Returns:
            True if trade opposes position (different signs)
        """
        position = self.get_position(ticker)
        if position is None:
            return False
        
        # Check if signs are different (opposing)
        return (position.quantity > 0 and trade_quantity < 0) or \
               (position.quantity < 0 and trade_quantity > 0)
    
    def clear_all_positions(self):
        """Clear all positions (for reset)"""
        self.positions.clear()
        logger.info("Cleared all positions")
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
    
    def print_all_positions(self):
        """Print all current positions for debugging"""
        if not self.positions:
            logger.info("No open positions")
        else:
            logger.info("Current positions:")
            for ticker, position in self.positions.items():
                logger.info(f"  {ticker}: {position.quantity} lots, strategy={position.strategy}")
    
    def __repr__(self):
        return f"PositionManager({len(self.positions)} positions)"
