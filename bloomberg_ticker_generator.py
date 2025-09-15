"""
Bloomberg Ticker Generator Module
Unified ticker generation to ensure consistency across all parsers
"""

from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Month codes for futures
MONTH_CODE = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

# Special index ticker mappings
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
    'FINNIFTY': {
        'futures_ticker': 'FNF',
        'options_ticker': 'FINNIFTY',
        'is_index': True
    },
    'FNF': {
        'futures_ticker': 'FNF',
        'options_ticker': 'FINNIFTY',
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

# Known index tickers (for detection)
INDEX_TICKERS = {
    'NZ', 'NBZ', 'NIFTY', 'BANKNIFTY', 'NF', 'NBF', 'FNF', 'FINNIFTY', 
    'MCN', 'MIDCPNIFTY', 'AF', 'AF1', 'NSEBANK', 'RNS', 'NMIDSELP'
}


def is_index_instrument(ticker: str, series: str = None) -> bool:
    """
    Determine if an instrument is an index
    
    Args:
        ticker: The ticker symbol
        series: Optional series/instrument type (e.g., FUTIDX, OPTIDX)
    
    Returns:
        True if it's an index instrument
    """
    ticker_upper = ticker.upper() if ticker else ""
    
    # Check series first (most reliable)
    if series:
        series_upper = series.upper()
        if 'IDX' in series_upper:  # FUTIDX, OPTIDX
            return True
    
    # Check if ticker is in known index list
    if ticker_upper in INDEX_TICKERS:
        return True
    
    # Check if ticker contains index keywords
    if 'NIFTY' in ticker_upper or ticker_upper.endswith('INDEX'):
        return True
    
    # Check special rules
    if ticker_upper in INDEX_TICKER_RULES:
        return INDEX_TICKER_RULES[ticker_upper].get('is_index', False)
    
    return False


def get_ticker_for_instrument(symbol: str, security_type: str, series: str = None) -> str:
    """
    Get the appropriate ticker based on instrument type and special rules
    
    Args:
        symbol: Original symbol
        security_type: Futures, Call, or Put
        series: Optional series/instrument type
    
    Returns:
        The ticker to use for Bloomberg format
    """
    symbol_upper = symbol.upper() if symbol else ""
    
    # Check for special index rules
    if symbol_upper in INDEX_TICKER_RULES:
        rule = INDEX_TICKER_RULES[symbol_upper]
        if security_type == 'Futures':
            return rule['futures_ticker']
        else:  # Options (Call or Put)
            return rule['options_ticker']
    
    # Default: return the symbol as-is
    return symbol


def generate_bloomberg_ticker(
    ticker: str,
    expiry: datetime,
    security_type: str,
    strike: float,
    series: str = None,
    original_symbol: str = None
) -> str:
    """
    Generate Bloomberg ticker format - UNIFIED VERSION
    
    Args:
        ticker: The Bloomberg ticker symbol
        expiry: Expiry date
        security_type: 'Futures', 'Call', or 'Put'
        strike: Strike price (0 for futures)
        series: Optional series/instrument type (FUTIDX, FUTSTK, etc.)
        original_symbol: Original symbol (for special mappings)
    
    Returns:
        Formatted Bloomberg ticker string
    """
    if not ticker:
        logger.warning("Empty ticker provided")
        return ""
    
    ticker = ticker.upper().strip()
    
    # Use original symbol for special mappings if provided
    check_symbol = original_symbol.upper() if original_symbol else ticker
    
    # Get the appropriate ticker based on rules
    final_ticker = get_ticker_for_instrument(check_symbol, security_type, series)
    
    # Determine if this is an index
    is_index = is_index_instrument(final_ticker, series)
    
    # Log for debugging
    logger.debug(f"Generating ticker: symbol={check_symbol}, ticker={final_ticker}, "
                f"type={security_type}, is_index={is_index}, series={series}")
    
    if security_type == 'Futures':
        # Futures format
        month_code = MONTH_CODE.get(expiry.month, "")
        year_code = str(expiry.year)[-1]  # Last digit of year
        
        if is_index:
            # Index futures: NZU5 Index (no = sign, space before Index)
            return f"{final_ticker}{month_code}{year_code} Index"
        else:
            # Stock futures: RIL=H5 IS Equity (= sign, space before IS)
            return f"{final_ticker}={month_code}{year_code} IS Equity"
    
    else:
        # Options format
        # Date format: MM/DD/YY
        date_str = expiry.strftime('%m/%d/%y')
        
        # Strike format: integer if whole number, otherwise decimal
        strike_str = str(int(strike)) if strike == int(strike) else str(strike)
        
        # Option type
        opt_type = 'C' if security_type == 'Call' else 'P'
        
        if is_index:
            # Index options: NIFTY 03/27/25 C21000 Index
            return f"{final_ticker} {date_str} {opt_type}{strike_str} Index"
        else:
            # Stock options: RIL IS 03/27/25 C1200 Equity
            return f"{final_ticker} IS {date_str} {opt_type}{strike_str} Equity"


def normalize_ticker_for_comparison(ticker: str) -> str:
    """
    Normalize a Bloomberg ticker for comparison purposes
    Useful for matching tickers that might have minor formatting differences
    
    Args:
        ticker: Bloomberg ticker string
    
    Returns:
        Normalized ticker string
    """
    if not ticker:
        return ""
    
    # Convert to uppercase and strip whitespace
    normalized = ticker.upper().strip()
    
    # Standardize multiple spaces to single space
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized
