"""
Input Parser Module - Updated with MIDCPNIFTY mapping
Uses same ticker generation logic as Trade_Parser for consistency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

@dataclass
class Position:
    """Represents a single position"""
    underlying_ticker: str
    bloomberg_ticker: str
    symbol: str
    expiry_date: datetime
    position_lots: float
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


class InputParser:
    """Parser that handles all three input formats"""
    
    def __init__(self, mapping_file: str = "futures mapping.csv"):
        self.mapping_file = mapping_file
        self.normalized_mappings = {}
        self.symbol_mappings = self._load_mappings()
        self.positions = []
        self.unmapped_symbols = []
        self.format_type = None
    
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
                    
                    # Handle underlying (column 3)
                    underlying = None
                    if len(row) > 2 and pd.notna(row.iloc[2]):
                        underlying_val = str(row.iloc[2]).strip()
                        if underlying_val and underlying_val.upper() != 'NAN':
                            underlying = underlying_val
                    
                    # If no underlying specified, create default
                    if not underlying:
                        # Special handling for known indices
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
            logger.info(f"Loaded {len(mappings)} symbol mappings for input parser")
                    
        except Exception as e:
            logger.error(f"Error loading mapping file: {e}")
            self.normalized_mappings = {}
            
        return mappings
    
    def _get_index_ticker(self, symbol: str, security_type: str) -> Optional[Dict]:
        """
        Get special ticker mapping for index futures vs options
        Returns None if no special rule applies
        """
        symbol_upper = symbol.upper()
        
        # Check if this symbol has special index rules
        if symbol_upper in INDEX_TICKER_RULES:
            rule = INDEX_TICKER_RULES[symbol_upper]
            
            # Return appropriate ticker based on security type
            if security_type == 'Futures':
                return {
                    'ticker': rule['futures_ticker'],
                    'underlying': rule.get('underlying', f"{symbol_upper} INDEX"),
                    'lot_size': 50 if 'NIFTY' in symbol_upper else 15  # Default lot sizes
                }
            else:  # Options (Call or Put)
                return {
                    'ticker': rule['options_ticker'],
                    'underlying': rule.get('underlying', f"{symbol_upper} INDEX"),
                    'lot_size': 50 if 'NIFTY' in symbol_upper else 15
                }
        
        return None
    
    def parse_file(self, file_path: str) -> List[Position]:
        """Parse input file and return positions"""
        df = None
        
        # Try reading the file with different passwords if it's an Excel file
        if file_path.endswith(('.xls', '.xlsx')):
            passwords = ['Aurigin2017', 'Aurigin2024', None]
            
            for pwd in passwords:
                try:
                    if pwd:
                        import msoffcrypto
                        import io
                        
                        decrypted = io.BytesIO()
                        with open(file_path, 'rb') as f:
                            file = msoffcrypto.OfficeFile(f)
                            file.load_key(password=pwd)
                            file.decrypt(decrypted)
                        
                        decrypted.seek(0)
                        df = pd.read_excel(decrypted, header=None)
                        logger.info(f"Successfully opened file with password")
                        break
                    else:
                        df = pd.read_excel(file_path, header=None)
                        break
                except Exception as e:
                    if 'encrypted' not in str(e).lower() and pwd is None:
                        logger.error(f"Error reading file: {e}")
                        raise
                    continue
            
            if df is None:
                import getpass
                user_pwd = getpass.getpass("Enter password for Excel file: ")
                try:
                    import msoffcrypto
                    import io
                    
                    decrypted = io.BytesIO()
                    with open(file_path, 'rb') as f:
                        file = msoffcrypto.OfficeFile(f)
                        file.load_key(password=user_pwd)
                        file.decrypt(decrypted)
                    
                    decrypted.seek(0)
                    df = pd.read_excel(decrypted, header=None)
                except Exception as e:
                    logger.error(f"Failed to open file with provided password: {e}")
                    return []
        else:
            df = pd.read_csv(file_path, header=None)
        
        if df is None:
            logger.error("Could not read input file")
            return []
        
        format_type = self._detect_format(df)
        logger.info(f"Detected format: {format_type}")
        
        self.format_type = format_type
        
        if format_type == 'BOD':
            return self._parse_bod(df)
        elif format_type == 'CONTRACT':
            return self._parse_contract(df)
        elif format_type == 'MS':
            return self._parse_ms(df)
        else:
            logger.error("Unknown file format")
            return []
    
    def _detect_format(self, df: pd.DataFrame) -> str:
        """Detect which format the file is in"""
        # MS format check
        if df.shape[1] >= 20:
            ms_pattern_found = False
            for i in range(min(50, len(df))):
                if pd.notna(df.iloc[i, 0]):
                    val = str(df.iloc[i, 0])
                    if (('FUTSTK' in val or 'OPTSTK' in val or 'FUTIDX' in val or 'OPTIDX' in val) 
                        and val.count('-') >= 4):
                        ms_pattern_found = True
                        break
            
            if ms_pattern_found:
                return 'MS'
        
        # CONTRACT format check
        if df.shape[1] >= 12:
            for i in range(min(20, len(df))):
                if len(df.iloc[i]) > 3 and pd.notna(df.iloc[i, 3]):
                    val = str(df.iloc[i, 3])
                    if ('FUTSTK' in val or 'OPTSTK' in val or 'FUTIDX' in val or 'OPTIDX' in val) and '-' in val:
                        return 'CONTRACT'
        
        # Default to BOD
        if df.shape[1] >= 16:
            return 'BOD'
        
        return 'UNKNOWN'
    
    def _parse_bod(self, df: pd.DataFrame) -> List[Position]:
        """Parse BOD format"""
        positions = []
        data_start = self._find_data_start_bod(df)
        
        for idx in range(data_start, len(df)):
            try:
                row = df.iloc[idx]
                if len(row) < 15:
                    continue
                
                symbol = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
                if not symbol:
                    continue
                
                series = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else 'EQ'
                expiry = pd.to_datetime(row.iloc[3]) if pd.notna(row.iloc[3]) else datetime.now() + timedelta(30)
                strike = float(row.iloc[4]) if pd.notna(row.iloc[4]) else 0.0
                option_type = str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else ''
                lot_size = int(row.iloc[6]) if pd.notna(row.iloc[6]) else 1
                
                try:
                    col13_val = float(row.iloc[13]) if pd.notna(row.iloc[13]) else 0.0
                except (ValueError, TypeError):
                    col13_val = 0.0
                
                try:
                    col14_val = float(row.iloc[14]) if pd.notna(row.iloc[14]) else 0.0
                except (ValueError, TypeError):
                    col14_val = 0.0
                
                position_lots = col13_val - col14_val
                
                if position_lots == 0:
                    continue
                
                series_upper = series.upper()
                if 'FUT' in series_upper:
                    inst_type = 'FF'
                elif 'OPT' in series_upper:
                    inst_type = option_type
                else:
                    inst_type = option_type if option_type else 'FF'
                
                position = self._create_position(
                    symbol, expiry, strike, inst_type, position_lots, lot_size, series
                )
                if position:
                    positions.append(position)
                    
            except Exception as e:
                logger.debug(f"Error parsing BOD row {idx}: {e}")
                
        logger.info(f"Parsed {len(positions)} positions from BOD format")
        return positions
    
    def _parse_contract(self, df: pd.DataFrame) -> List[Position]:
        """Parse Contract CSV format"""
        positions = []
        
        for idx in range(len(df)):
            try:
                row = df.iloc[idx]
                if len(row) < 12:
                    continue
                
                if idx == 0:
                    if pd.notna(row[5]) and str(row[5]).strip().lower() == 'lot size':
                        continue
                
                contract_id = str(row[3]).strip() if pd.notna(row[3]) else ""
                
                if not contract_id or not ('FUTSTK' in contract_id or 'OPTSTK' in contract_id 
                                          or 'FUTIDX' in contract_id or 'OPTIDX' in contract_id):
                    continue
                
                try:
                    lot_size = int(float(row[5])) if pd.notna(row[5]) else 0
                except (ValueError, TypeError):
                    lot_size = 0
                
                try:
                    position_lots = float(row[10]) if pd.notna(row[10]) else 0
                except (ValueError, TypeError):
                    position_lots = 0
                
                if position_lots == 0:
                    continue
                
                parsed = self._parse_contract_id(contract_id)
                if parsed:
                    position = self._create_position(
                        parsed['symbol'], parsed['expiry'], parsed['strike'],
                        parsed['inst_type'], position_lots, lot_size, parsed['series']
                    )
                    if position:
                        positions.append(position)
                        
            except Exception as e:
                logger.debug(f"Error parsing CONTRACT row {idx}: {e}")
        
        logger.info(f"Parsed {len(positions)} positions from CONTRACT format")
        return positions
    
    def _parse_ms(self, df: pd.DataFrame) -> List[Position]:
        """Parse MS Position format"""
        positions = []
        
        for idx in range(len(df)):
            try:
                row = df.iloc[idx]
                if len(row) < 21:
                    continue
                
                contract_id = str(row[0]).strip() if pd.notna(row[0]) else ""
                
                if not contract_id or '-' not in contract_id:
                    continue
                
                if any(keyword in contract_id.lower() for keyword in ['total', 'summary', 'net', 'mtm']):
                    continue
                
                try:
                    col20_val = float(row[19]) if pd.notna(row[19]) else 0.0
                except (ValueError, TypeError):
                    col20_val = 0.0
                
                try:
                    col21_val = float(row[20]) if pd.notna(row[20]) else 0.0
                except (ValueError, TypeError):
                    col21_val = 0.0
                
                position_lots = col20_val - col21_val
                
                if position_lots == 0:
                    continue
                
                parsed = self._parse_contract_id(contract_id)
                if parsed:
                    position = self._create_position(
                        parsed['symbol'], parsed['expiry'], parsed['strike'],
                        parsed['inst_type'], position_lots, None, parsed['series']
                    )
                    if position:
                        positions.append(position)
                        
            except Exception as e:
                logger.debug(f"Could not parse MS row {idx}: {e}")
        
        logger.info(f"Successfully parsed {len(positions)} positions from MS format")
        return positions
    
    def _find_data_start_bod(self, df: pd.DataFrame) -> int:
        """Find where data starts in BOD format"""
        for i in range(min(100, len(df))):
            if len(df.iloc[i]) < 15:
                continue
            
            col5_val = str(df.iloc[i, 4]).strip() if pd.notna(df.iloc[i, 4]) else ""
            if any(word in col5_val.lower() for word in ['strike', 'price', 'column', 'header']):
                continue
            
            try:
                if pd.notna(df.iloc[i, 4]):
                    float(df.iloc[i, 4])
                if pd.notna(df.iloc[i, 13]) or pd.notna(df.iloc[i, 14]):
                    float(df.iloc[i, 13] if pd.notna(df.iloc[i, 13]) else 0)
                    float(df.iloc[i, 14] if pd.notna(df.iloc[i, 14]) else 0)
                return i
            except:
                continue
        return 0
    
    def _parse_contract_id(self, contract_id: str) -> Optional[Dict]:
        """Parse contract ID string"""
        try:
            contract_id = contract_id.strip()
            if contract_id.endswith(' -0'):
                contract_id = contract_id[:-3]
            
            parts = contract_id.split('-')
            parts = [p.strip() for p in parts]
            
            if len(parts) < 5:
                return None
            
            series = parts[0]
            strike_str = parts[-1].replace(',', '')
            inst_type = parts[-2]
            expiry_str = parts[-3]
            symbol_parts = parts[1:-3]
            symbol = '-'.join(symbol_parts) if symbol_parts else parts[1]
            
            expiry = self._parse_date(expiry_str)
            strike = float(strike_str)
            
            return {
                'series': series,
                'symbol': symbol,
                'expiry': expiry,
                'inst_type': inst_type,
                'strike': strike
            }
        except:
            return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string - MATCHING TRADE PARSER"""
        date_str = str(date_str).strip()
        
        # Try different date formats - SAME AS TRADE PARSER
        formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
            '%m/%d/%y', '%m-%d-%y', '%m.%d.%y',
            '%d-%b-%Y', '%d-%b-%y',  # For formats like 26-Sep-2025
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # Month map for manual parsing
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        # Try manual parsing for DDMMMYYYY format
        date_str_upper = date_str.upper()
        match = re.match(r'(\d{1,2})([A-Z]{3})(\d{4})', date_str_upper.replace('-', ''))
        if match:
            day = int(match.group(1))
            month = month_map.get(match.group(2), 0)
            year = int(match.group(3))
            if month:
                return datetime(year, month, day)
        
        # Try pandas parser as fallback
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    def _create_position(self, symbol: str, expiry: datetime, strike: float,
                        inst_type: str, position_lots: float, lot_size: Optional[int],
                        series: str) -> Optional[Position]:
        """Create Position object from parsed data - MATCHING TRADE PARSER LOGIC"""
        symbol_normalized = symbol.strip().upper()
        
        # Check for special index ticker rules first - SAME AS TRADE PARSER
        security_type = self._determine_security_type(inst_type, series)
        if not security_type:
            return None
        
        special_mapping = self._get_index_ticker(symbol_normalized, security_type)
        
        if special_mapping:
            # Use special index rules
            mapping = special_mapping
            mapping['original_symbol'] = symbol
            # Override lot size with the one from position file if available
            if lot_size:
                mapping['lot_size'] = lot_size
        else:
            # Get regular mapping from file
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
                    'position_lots': position_lots
                })
                return None
        
        # Generate Bloomberg ticker - MATCHING TRADE PARSER
        bloomberg_ticker = self._generate_bloomberg_ticker(
            mapping['ticker'], expiry, security_type, strike, series
        )
        
        # Handle lot_size
        if lot_size is not None and lot_size > 0:
            final_lot_size = lot_size
        else:
            final_lot_size = mapping.get('lot_size', 1)
            if not final_lot_size or final_lot_size == 0:
                final_lot_size = 1
        
        logger.debug(f"Created position: {symbol} -> {bloomberg_ticker}")
        
        return Position(
            underlying_ticker=mapping.get('underlying', f"{mapping['ticker']} IS Equity"),
            bloomberg_ticker=bloomberg_ticker,
            symbol=symbol,
            expiry_date=expiry,
            position_lots=position_lots,
            security_type=security_type,
            strike_price=strike,
            lot_size=final_lot_size
        )
    
    def _determine_security_type(self, inst_type: str, series: str) -> Optional[str]:
        """Determine security type from inst_type and series"""
        inst_type = inst_type.upper()
        series_upper = series.upper() if series else ''
        
        if inst_type == 'FF' or 'FUT' in inst_type or 'FUT' in series_upper:
            return 'Futures'
        elif inst_type in ['CE', 'C', 'CALL']:
            return 'Call'
        elif inst_type in ['PE', 'P', 'PUT']:
            return 'Put'
        else:
            logger.debug(f"Could not determine security type for inst_type={inst_type}, series={series}")
            return None
    
    def _generate_bloomberg_ticker(self, ticker: str, expiry: datetime,
                                  security_type: str, strike: float, series: str = None) -> str:
        """Generate Bloomberg ticker - MATCHING TRADE PARSER EXACTLY"""
        ticker_upper = ticker.upper()
        
        # Check if this is an index based on ticker or series
        is_index = False
        if series:
            series_upper = series.upper()
            if 'IDX' in series_upper:  # FUTIDX, OPTIDX
                is_index = True
        
        # Also check ticker itself - UPDATED WITH NEW TICKERS
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
            # Options format
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
