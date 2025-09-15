# Trade Strategy Processing System

A comprehensive Python application for processing trading positions and trades, automatically assigning FULO/FUSH strategies based on position directions and trade flows.

## 🎯 Overview

This system processes position and trade files from various formats, tracks position changes, assigns appropriate trading strategies (FULO/FUSH), and handles complex scenarios like position flips and split trades. It provides a user-friendly Streamlit interface for easy operation.

## ✨ Features

- **Multi-format Support**: Handles BOD, Contract, and MS position file formats
- **Strategy Assignment**: Automatically assigns FULO (long) or FUSH (short) strategies
- **Trade Splitting**: Intelligently splits trades when positions flip direction
- **Position Tracking**: Maintains accurate position state throughout processing
- **Missing Mapping Detection**: Identifies and reports unmapped symbols
- **Bloomberg Ticker Generation**: Creates properly formatted Bloomberg tickers
- **Excel Output**: Generates comprehensive reports in CSV and Excel formats

## 📋 Requirements

```bash
pandas>=1.3.0
numpy>=1.20.0
streamlit>=1.20.0
openpyxl>=3.0.0
msoffcrypto-tool>=5.0.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trade-strategy-processor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the mapping file:
   - Place `futures mapping.csv` in the project root directory

## 💻 Usage

### Running the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Upload Position File**: Select your position file (BOD, Contract, or MS format)
2. **Upload Trade File**: Select your trade file (MS format)
3. **Mapping File**: Use the default `futures mapping.csv` or upload a custom one
4. **Process**: Click "Process Trades" to run the analysis
5. **Review Results**: Check the processed trades and download output files

## 📁 File Formats

### Position File Formats

#### BOD Format
- Columns: Symbol, Series, Expiry, Strike, Option Type, Lot Size, Buy Qty (col 13), Sell Qty (col 14)
- Position = Buy Qty - Sell Qty

#### Contract Format
- Column 3: Contract ID (e.g., "FUTSTK-RELIANCE-26SEP2024-FF-0")
- Column 5: Lot Size
- Column 10: Net Position

#### MS Format
- Column 0: Contract ID
- Columns 19-20: Buy/Sell quantities
- 21+ columns expected

### Trade File Format (MS)
- 14 columns expected
- Column 4: Instrument type (OPTSTK/OPTIDX/FUTSTK/FUTIDX)
- Column 5: Symbol
- Column 6: Expiry Date
- Column 7: Lot Size
- Column 8: Strike Price
- Column 9: Option Type (CE/PE)
- Column 10: Buy/Sell
- Column 11: Quantity
- Column 12: Lots Traded

### Mapping File Format
```csv
Symbol,Ticker,Underlying,Exchange,Lot_Size
RELIANCE,RELIANCE,RELIANCE IS Equity,IS Equity,250
NIFTY,NZ,NIFTY INDEX,Index,50
BANKNIFTY,AF1,BANKNIFTY INDEX,Index,15
MIDCPNIFTY,RNS,MIDCPNIFTY INDEX,Index,50
```

## 🎯 Strategy Assignment Rules

### FULO (Long Strategy)
- Long Futures positions
- Long Call positions  
- Short Put positions

### FUSH (Short Strategy)
- Short Futures positions
- Short Call positions
- Long Put positions

### Key Principles
1. **New Positions**: Strategy determined by trade direction and instrument type
2. **Closing Positions**: Inherit the strategy of the position being closed
3. **Split Trades**: When a trade flips a position:
   - First split closes existing position (inherits strategy)
   - Second split opens new position (gets new strategy)

## 📊 Output Files

The system generates multiple output files:

1. **Parsed Trades** (`output_1_parsed_trades_*.csv`)
   - Original trades as parsed from input

2. **Starting Positions** (`output_2_starting_positions_*.csv`)
   - Initial position state before trades

3. **Processed Trades** (`output_3_processed_trades_*.csv`)
   - Main output with strategy assignments
   - Includes Split? and Opposite? flags
   - Available in both CSV and Excel formats

4. **Final Positions** (`output_4_final_positions_*.csv`)
   - Position state after all trades

5. **Missing Mappings** (`MISSING_MAPPINGS_*.csv`)
   - Symbols that couldn't be mapped
   - Template file for easy addition to mapping

6. **Summary Report** (`summary_report_*.txt`)
   - Processing statistics and overview

## 🔧 Project Structure

```
trade-strategy-processor/
├── streamlit_app.py           # Main Streamlit application
├── input_parser.py            # Position file parser
├── Trade_Parser.py            # Trade file parser
├── position_manager.py        # Position tracking and management
├── trade_processor.py         # Trade processing and strategy assignment
├── output_generator.py        # Output file generation
├── bloomberg_ticker_generator.py  # Bloomberg ticker formatting
├── futures mapping.csv        # Symbol to ticker mappings
└── output/                    # Generated output files
```

## 🔍 Special Index Mappings

The system includes special handling for index instruments:

| Symbol | Futures Ticker | Options Ticker |
|--------|---------------|----------------|
| NIFTY | NZ | NIFTY |
| BANKNIFTY | AF1 | NSEBANK |
| MIDCPNIFTY | RNS | NMIDSELP |
| FINNIFTY | FNF | FINNIFTY |

## ⚠️ Important Notes

1. **Position Flips**: When trades cause positions to flip (long to short or vice versa), the system automatically splits the trade and assigns appropriate strategies

2. **QTY Calculation**: Quantities are always calculated as `Lots × Lot_Size` to avoid decimals

3. **Missing Mappings**: Unmapped symbols are skipped but reported in the missing mappings file

4. **Password-Protected Files**: The system can handle Excel files with passwords (Aurigin2017, Aurigin2024)

## 🐛 Troubleshooting

### Common Issues

1. **"No positions found"**
   - Check file format matches expected structure
   - Verify data starts at correct row
   - Ensure position quantities are non-zero

2. **"No trades found"**
   - Verify trade file is in MS format (14 columns)
   - Check instrument types are valid (FUTSTK, OPTSTK, etc.)

3. **Missing mappings**
   - Download the MISSING_MAPPINGS file
   - Add missing symbols to futures mapping.csv
   - Use the MAPPING_TEMPLATE file for correct format

4. **Strategy assignment issues**
   - Verify position tracking is working correctly
   - Check trade sequence and directions
   - Review split trade logic for flips

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Name/Team]

## 📧 Support

For issues or questions, please contact [your contact information]

---

*Last Updated: [Date]*
