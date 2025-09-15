"""
Trade Parser Module - Updated with MIDCPNIFTY mapping
NO AGGREGATION - Each trade line processed individually
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Define Position class locally
@dataclass
class Position:
    """Represents a single trade (not aggregated position)"""
    underlying_ticker: str
    bloomberg_ticker: str
    symbol: str
    expiry_date: datetime
    position_lots: float  # This is the trade quantity with sign
    security_type: str  # Futures, Call, Put
    strike_price: float
    lot_size: int
    
    @property
    def is_future(self) -> bool:
        return self.security_type == 'Futures'
    
    @property
    def is_call(self) -> bool:
        return self.security_type == 'Call'
    
    @property
    def is_put(self) -> bool:
        return self.security_type == 'Put'

# Constants
MONTH_CODE = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

# Special index ticker mappings - UPDATED WITH MIDCPNIFTY
INDEX_TICKER_RULES = {
    'NIFTY': {
        'futures_ticker': 'NZ',
        'options_ticker': 'NIFTY',
        'is_index': True
    },
    'NZ': {
        'futures_ticker': 'NZ',
        'options_ticker': 'NIFTY',
        'is_index': True
    },
    'BANKNIFTY': {
        'futures_ticker': 'AF1',
        'options_ticker': 'NSEBANK',
        'is_index': True
    },
    'AF1': {
        'futures_ticker': 'AF1',
        'options_ticker': 'NSEBANK',
        'is_index': True
    },
    'AF': {
        'futures_ticker': 'AF1',
        'options_ticker': 'NSEBANK',
        'is_index': True
    },
    'NSEBANK': {
        'futures_ticker': 'AF1',
        'options_ticker': 'NSEBANK',
        'is_index': True
    },
    'MIDCPNIFTY': {
        'futures_ticker': 'RNS',
        'options_ticker': 'NMIDSELP',
        'is_index': True
    },
    'RNS': {
        'futures_ticker': 'RNS',
        'options_ticker': 'NMIDSELP',
        'is_index': True
    },
    'NMIDSELP': {
        'futures_ticker': 'RNS',
        'options_ticker': 'NMIDSELP',
        'is_index': True
    },
    'MCN': {
        'futures_ticker': 'RNS',
        'options_ticker': 'NMIDSELP',
        'is_index': True
    }
}


class TradeParser:
    """Parser for trade files - NO AGGREGATION VERSION"""
    
    def __init__(self, mapping_file: str = "futures mapping.csv"):
        self.mapping_file = mapping_file
        self.symbol_mappings = self._load_mappings()
        self.trades = []
        self.format_type = None
        self.unmapped_symbols = []
        
    def _load_mappings(self) -> Dict:
        """Load symbol mappings from CSV"""
        mappings = {}
        normalized_mappings = {}
        
        try:
            df = pd.read_csv(self.mapping_file)
            for idx, row in df.iterrows():
                if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    symbol = str(row.iloc[0]).strip()
                    ticker = str(row.iloc[1]).strip()
                    
                    underlying = None
                    if len(row) > 2 and pd.notna(row.iloc[2]):
                        underlying_val = str(row.iloc[2]).strip()
                        if underlying_val and underlying_val.upper() != 'NAN':
                            underlying = underlying_val
                    
                    if not underlying:
                        if symbol.upper() in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
                            underlying = f"{symbol.upper()} INDEX"
                        else:
                            underlying = f"{ticker} IS Equity"
                    
                    lot_size = 1
                    if len(row) > 4 and pd.notna(row.iloc[4]):
                        try:
                            lot_size = int(float(str(row.iloc[4]).strip()))
                        except (ValueError, TypeError):
                            lot_size = 1
                    
                    mapping = {
                        'ticker': ticker,
                        'underlying': underlying,
                        'lot_size': lot_size,
                        'original_symbol': symbol
                    }
                    mappings[symbol] = mapping
                    normalized_mappings[symbol.upper()] = mapping
            
            self.normalized_mappings = normalized_mappings
            logger.info(f"Loaded {len(mappings)} symbol mappings for trade parser")
            
        except Exception as e:
            logger.error(f"Error loading mapping file: {e}")
            self.normalized_mappings = {}
            
        return mappings
    
    def _get_index_ticker(self, symbol: str, security_type: str) -> Optional[Dict]:
        """Get special ticker mapping for index futures vs options"""
        symbol_upper = symbol.upper()
        
        if symbol_upper in INDEX_TICKER_RULES:
            rule = INDEX_TICKER_RULES[symbol_upper]
            
            if security_type == 'Futures':
                return {
                    'ticker': rule['futures_ticker'],
                    'underlying': rule.get('underlying', f"{symbol_upper} INDEX"),
                    'lot_size': 50 if 'NIFTY' in symbol_upper else 15
                }
            else:  # Options
                return {
                    'ticker': rule['options_ticker'],
                    'underlying': rule.get('underlying', f"{symbol_upper} INDEX"),
                    'lot_size': 50 if 'NIFTY' in symbol_upper else 15
                }
        
        return None
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """Detect if it's MS or GS trade format"""
        if df.shape[1] == 14:
            try:
                col4_vals = df.iloc[:, 4].dropna().astype(str).str.upper()
                if any(val in ['OPTSTK', 'OPTIDX', 'FUTSTK', 'FUTIDX'] for val in col4_vals):
                    logger.info("Detected MS trade format (14 columns)")
                    return 'MS'
            except:
                pass
        
        logger.info("Detected GS trade format (non-MS structure)")
        return 'GS'
    
    def parse_trade_file(self, file_path: str) -> List[Position]:
        """Parse trade file - RETURN EACH TRADE LINE INDIVIDUALLY"""
        try:
            # Read file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=None if self._has_no_header(file_path) else 0)
            else:
                df = pd.read_excel(file_path, header=None if self._has_no_header(file_path) else 0)
            
            self.format_type = self.detect_format(df)
            
            if self.format_type == 'MS':
                return self._parse_ms_trades_sequential(df)
            else:
                return self._parse_gs_trades(df)
        except Exception as e:
            logger.error(f"Error in parse_trade_file: {e}")
            return []
    
    def _has_no_header(self, file_path: str) -> bool:
        """Check if file has headers or not"""
        try:
            if file_path.endswith('.csv'):
                first_row = pd.read_csv(file_path, nrows=1)
            else:
                first_row = pd.read_excel(file_path, nrows=1)
            
            first_vals = first_row.iloc[0].astype(str)
            header_keywords = ['symbol', 'expiry', 'strike', 'option', 'instr', 'qty', 'price']
            if any(keyword in str(first_vals).lower() for keyword in header_keywords):
                return False
            return True
        except:
            return True
    
    def _parse_ms_trades_sequential(self, df: pd.DataFrame) -> List[Position]:
        """
        Parse MS format trade file - EACH LINE INDIVIDUALLY
        NO AGGREGATION - Returns trades in order they appear
        """
        trades = []  # List of individual trades, not aggregated
        
        # Check if first row is headers
        start_row = 0
        if df.shape[0] > 0:
            first_row = df.iloc[0]
            if any(str(val).lower() in ['symbol', 'expiry', 'strike'] for val in first_row if pd.notna(val)):
                start_row = 1
                logger.info("Skipping header row in trade file")
        
        # Process each row as a separate trade
        for idx in range(start_row, len(df)):
            try:
                row = df.iloc[idx]
                
                if len(row) < 14:
                    continue
                
                # MS format columns:
                # 4: Instr (OPTSTK/OPTIDX/FUTSTK/FUTIDX)
                # 5: Symbol
                # 6: Expiry Dt
                # 7: Lot Size
                # 8: Strike Price
                # 9: Option Type (CE/PE)
                # 10: B/S (Buy/Sell)
                # 11: Qty
                # 12: Lots Traded
                
                instr = str(row.iloc[4]).strip().upper() if pd.notna(row.iloc[4]) else ""
                if instr not in ['OPTSTK', 'OPTIDX', 'FUTSTK', 'FUTIDX']:
                    continue
                
                symbol = str(row.iloc[5]).strip().upper() if pd.notna(row.iloc[5]) else ""
                if not symbol:
                    continue
                    
                expiry_str = str(row.iloc[6]).strip() if pd.notna(row.iloc[6]) else ""
                lot_size = int(float(row.iloc[7])) if pd.notna(row.iloc[7]) else 1
                strike = float(row.iloc[8]) if pd.notna(row.iloc[8]) else 0
                option_type = str(row.iloc[9]).strip().upper() if pd.notna(row.iloc[9]) else ""
                side = str(row.iloc[10]).strip().upper() if pd.notna(row.iloc[10]) else ""
                lots = float(row.iloc[12]) if pd.notna(row.iloc[12]) else 0
                
                if lots == 0:
                    continue
                
                # Parse expiry date
                expiry = self._parse_date(expiry_str)
                if not expiry:
                    continue
                
                # Determine security type
                if 'FUT' in instr:
                    security_type = 'Futures'
                    strike = 0
                elif option_type in ['CE', 'C', 'CALL']:
                    security_type = 'Call'
                elif option_type in ['PE', 'P', 'PUT']:
                    security_type = 'Put'
                else:
                    continue
                
                # Determine trade direction
                if side.startswith('B'):
                    trade_lots = lots  # Positive for buy
                elif side.startswith('S'):
                    trade_lots = -lots  # Negative for sell
                else:
                    continue
                
                # Get mapping - check special rules first
                special_mapping = self._get_index_ticker(symbol, security_type)
                
                if special_mapping:
                    mapping = special_mapping
                    mapping['original_symbol'] = symbol
                    mapping['lot_size'] = lot_size  # Use trade file lot size
                else:
                    # Get regular mapping
                    symbol_normalized = symbol.strip().upper()
                    mapping = None
                    if symbol in self.symbol_mappings:
                        mapping = self.symbol_mappings[symbol]
                    elif symbol_normalized in self.normalized_mappings:
                        mapping = self.normalized_mappings[symbol_normalized]
                    
                    if not mapping:
                        logger.warning(f"No mapping found for symbol: {symbol}")
                        self.unmapped_symbols.append({
                            'symbol': symbol,
                            'expiry': expiry,
                            'position_lots': trade_lots
                        })
                        continue
                
                # Generate Bloomberg ticker
                bloomberg_ticker = self._generate_bloomberg_ticker(
                    mapping['ticker'], expiry, security_type, strike, instr
                )
                
                # Create trade object (NOT aggregated position)
                trade = Position(
                    underlying_ticker=mapping.get('underlying', f"{mapping['ticker']} IS Equity"),
                    bloomberg_ticker=bloomberg_ticker,
                    symbol=symbol,
                    expiry_date=expiry,
                    position_lots=trade_lots,  # Individual trade quantity with sign
                    security_type=security_type,
                    strike_price=strike,
                    lot_size=lot_size
                )
                
                trades.append(trade)
                logger.debug(f"Trade {idx}: {symbol} {side} {lots} lots -> {bloomberg_ticker}")
                    
            except Exception as e:
                logger.debug(f"Error parsing MS trade row {idx}: {e}")
        
        logger.info(f"Parsed {len(trades)} individual trades (not aggregated)")
        return trades
    
    def _parse_gs_trades(self, df: pd.DataFrame) -> List[Position]:
        """Parse GS format trade file"""
        logger.warning("GS format parsing not yet implemented")
        return []
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string in various formats"""
        date_str = str(date_str).strip()
        
        formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
            '%m/%d/%y', '%m-%d-%y', '%m.%d.%y',
            '%d-%b-%Y', '%d-%b-%y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    def _generate_bloomberg_ticker(self, ticker: str, expiry: datetime,
                                  security_type: str, strike: float,
                                  series: str = None) -> str:
        """Generate Bloomberg ticker format"""
        ticker_upper = ticker.upper()
        
        # Check if index - UPDATED WITH NEW TICKERS
        is_index = False
        if series:
            series_upper = series.upper()
            if 'IDX' in series_upper:
                is_index = True
        
        if ticker_upper in ['NZ', 'NBZ', 'NIFTY', 'BANKNIFTY', 'AF1', 'NSEBANK', 'RNS', 'NMIDSELP', 'MCN', 'MIDCPNIFTY'] or 'NIFTY' in ticker_upper:
            is_index = True
        
        if security_type == 'Futures':
            month_code = MONTH_CODE.get(expiry.month, "")
            year_code = str(expiry.year)[-1]
            
            if is_index:
                return f"{ticker}{month_code}{year_code} Index"
            else:
                return f"{ticker}={month_code}{year_code} IS Equity"
        else:
            date_str = expiry.strftime('%m/%d/%y')
            strike_str = str(int(strike)) if strike == int(strike) else str(strike)
            
            if is_index:
                if security_type == 'Call':
                    return f"{ticker} {date_str} C{strike_str} Index"
                else:
                    return f"{ticker} {date_str} P{strike_str} Index"
            else:
                if security_type == 'Call':
                    return f"{ticker} IS {date_str} C{strike_str} Equity"
                else:
                    return f"{ticker} IS {date_str} P{strike_str} Equity"
