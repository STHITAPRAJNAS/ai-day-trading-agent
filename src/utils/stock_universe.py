from typing import List, Dict
import requests
import logging

logger = logging.getLogger(__name__)

# Pre-defined stock universes for testing and development
STOCK_UNIVERSES = {
    "MEGA_CAP": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "UNH", "JNJ",
        "XOM", "JPM", "V", "PG", "HD", "CVX", "MA", "BAC", "ABBV", "PFE"
    ],

    "TECH_GROWTH": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "CRM", "ADBE", "NFLX",
        "PYPL", "INTC", "AMD", "ORCL", "IBM", "QCOM", "TXN", "AVGO", "CSCO", "SNOW"
    ],

    "DIVIDEND_ARISTOCRATS": [
        "JNJ", "PG", "KO", "PEP", "WMT", "MCD", "VZ", "T", "XOM", "CVX",
        "MMM", "CAT", "IBM", "GS", "HD", "TGT", "LOW", "ABBV", "MRK", "PFE"
    ],

    "SP500_SAMPLE": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "UNH", "JNJ",
        "XOM", "JPM", "V", "PG", "HD", "CVX", "MA", "BAC", "ABBV", "PFE",
        "KO", "PEP", "NFLX", "ADBE", "CRM", "PYPL", "INTC", "AMD", "ORCL", "IBM",
        "WMT", "MCD", "VZ", "T", "DIS", "CMCSA", "NKE", "TMO", "LLY", "ACN",
        "COST", "AVGO", "TXN", "HON", "UPS", "QCOM", "SBUX", "MDT", "LOW", "AMT"
    ],

    "SMALL_CAP_GROWTH": [
        "CRWD", "ZM", "DOCU", "SNOW", "PLTR", "ROKU", "TWLO", "SQ", "SHOP", "DDOG",
        "OKTA", "ZS", "ESTC", "FSLY", "NET", "COUP", "BILL", "FROG", "AI", "PATH"
    ],

    "ENERGY_SECTOR": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "KMI", "OKE",
        "WMB", "BKR", "HAL", "DVN", "FANG", "APA", "MRO", "CNX", "RRC", "AR"
    ],

    "HEALTHCARE": [
        "JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "MRK", "DHR", "BMY", "LLY",
        "GILD", "AMGN", "CVS", "CI", "ANTM", "HUM", "ISRG", "ZTS", "SYK", "BSX"
    ],

    "FINANCIAL": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
        "AXP", "BLK", "SCHW", "CB", "ICE", "CME", "SPGI", "MCO", "AON", "MMC"
    ]
}

def get_stock_universe(universe_name: str = "SP500_SAMPLE") -> List[str]:
    """
    Get a predefined stock universe for analysis

    Args:
        universe_name: Name of the stock universe to retrieve

    Returns:
        List of stock symbols
    """
    universe_name = universe_name.upper()

    if universe_name in STOCK_UNIVERSES:
        logger.info(f"Retrieved {universe_name} universe with {len(STOCK_UNIVERSES[universe_name])} symbols")
        return STOCK_UNIVERSES[universe_name].copy()

    logger.warning(f"Unknown universe '{universe_name}', falling back to SP500_SAMPLE")
    return STOCK_UNIVERSES["SP500_SAMPLE"].copy()

def get_sp500_symbols() -> List[str]:
    """
    Fetch current S&P 500 symbols from Wikipedia (for production use)
    Note: This requires internet connection and may be rate limited
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()

        # Clean symbols (remove dots, etc.)
        cleaned_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str):
                # Replace dots with dashes for Yahoo Finance compatibility
                cleaned_symbol = symbol.replace('.', '-')
                cleaned_symbols.append(cleaned_symbol)

        logger.info(f"Fetched {len(cleaned_symbols)} S&P 500 symbols from Wikipedia")
        return cleaned_symbols

    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 symbols: {str(e)}")
        logger.info("Falling back to predefined SP500_SAMPLE")
        return get_stock_universe("SP500_SAMPLE")

def get_nasdaq100_symbols() -> List[str]:
    """
    Get NASDAQ 100 symbols (simplified version)
    In production, this would fetch from a reliable data source
    """
    nasdaq100 = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "PEP", "COST",
        "ADBE", "CMCSA", "NFLX", "INTC", "TXN", "QCOM", "TMUS", "AVGO", "HON", "AMGN",
        "SBUX", "AMD", "GILD", "BKNG", "MDLZ", "ADP", "ISRG", "CSX", "REGN", "FISV",
        "ATVI", "BIIB", "CHTR", "VRTX", "ILMN", "MU", "AMAT", "LRCX", "EXC", "WBA",
        "EA", "KHC", "CTSH", "DLTR", "EBAY", "VRSK", "XEL", "FAST", "PAYX", "CTAS"
    ]
    return nasdaq100

def filter_universe_by_criteria(
    symbols: List[str],
    min_market_cap: float = None,
    max_market_cap: float = None,
    min_volume: float = None,
    sectors: List[str] = None
) -> List[str]:
    """
    Filter stock universe based on various criteria
    Note: This is a simplified version. Production would use real-time data

    Args:
        symbols: List of symbols to filter
        min_market_cap: Minimum market cap in billions
        max_market_cap: Maximum market cap in billions
        min_volume: Minimum average daily volume
        sectors: List of sectors to include

    Returns:
        Filtered list of symbols
    """
    # For now, return the original list
    # In production, this would integrate with financial data APIs
    # to filter based on real-time criteria

    logger.info(f"Filtering universe: {len(symbols)} symbols (criteria filtering not implemented)")
    return symbols

def get_sector_symbols(sector: str) -> List[str]:
    """
    Get symbols for a specific sector

    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare', 'Energy')

    Returns:
        List of symbols in that sector
    """
    sector_mapping = {
        "TECHNOLOGY": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ADBE", "CRM", "ORCL", "INTC"],
        "HEALTHCARE": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "MRK", "DHR", "BMY", "LLY"],
        "FINANCIAL": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"],
        "ENERGY": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "KMI", "OKE"],
        "CONSUMER_DISCRETIONARY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "DIS", "BKNG"],
        "CONSUMER_STAPLES": ["PG", "KO", "PEP", "WMT", "COST", "CL", "GIS", "KHC", "HSY", "K"],
        "INDUSTRIALS": ["BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "FDX", "CSX"],
        "UTILITIES": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ES"],
        "MATERIALS": ["LIN", "APD", "SHW", "FCX", "NUE", "DOW", "DD", "PPG", "IFF", "ALB"],
        "REAL_ESTATE": ["AMT", "CCI", "PLD", "EQIX", "PSA", "WY", "DLR", "O", "SBAC", "AVB"],
        "TELECOMMUNICATIONS": ["VZ", "T", "TMUS", "CHTR", "CMCSA", "DIS", "NFLX", "FOXA", "PARA", "WBD"]
    }

    sector_upper = sector.upper()
    if sector_upper in sector_mapping:
        return sector_mapping[sector_upper].copy()

    logger.warning(f"Unknown sector '{sector}', returning empty list")
    return []

def get_universe_info() -> Dict[str, Dict]:
    """
    Get information about available stock universes

    Returns:
        Dictionary with universe names and their descriptions
    """
    return {
        "MEGA_CAP": {
            "description": "Top 20 largest market cap stocks",
            "count": len(STOCK_UNIVERSES["MEGA_CAP"]),
            "focus": "Large, stable companies"
        },
        "TECH_GROWTH": {
            "description": "Technology growth stocks",
            "count": len(STOCK_UNIVERSES["TECH_GROWTH"]),
            "focus": "High-growth technology companies"
        },
        "DIVIDEND_ARISTOCRATS": {
            "description": "Stocks with consistent dividend growth",
            "count": len(STOCK_UNIVERSES["DIVIDEND_ARISTOCRATS"]),
            "focus": "Income-generating stocks"
        },
        "SP500_SAMPLE": {
            "description": "Representative sample of S&P 500 stocks",
            "count": len(STOCK_UNIVERSES["SP500_SAMPLE"]),
            "focus": "Diversified large-cap stocks"
        },
        "SMALL_CAP_GROWTH": {
            "description": "High-growth small to mid-cap stocks",
            "count": len(STOCK_UNIVERSES["SMALL_CAP_GROWTH"]),
            "focus": "Emerging growth companies"
        },
        "ENERGY_SECTOR": {
            "description": "Energy sector stocks",
            "count": len(STOCK_UNIVERSES["ENERGY_SECTOR"]),
            "focus": "Oil, gas, and energy companies"
        },
        "HEALTHCARE": {
            "description": "Healthcare and pharmaceutical stocks",
            "count": len(STOCK_UNIVERSES["HEALTHCARE"]),
            "focus": "Healthcare industry leaders"
        },
        "FINANCIAL": {
            "description": "Financial services stocks",
            "count": len(STOCK_UNIVERSES["FINANCIAL"]),
            "focus": "Banks and financial institutions"
        }
    }