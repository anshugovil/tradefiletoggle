# tradefiletoggle
Trade file strategy assignment and toggle remover
# Trade Strategy Processing System

A modular Python system for processing trades against positions to assign trading strategies based on position directions. The system handles futures, options (calls and puts) with special logic for put options' inverted directional exposure.

## 🎯 Key Features

- **Strategy Assignment**: Automatically assigns FULO (long exposure) or FUSH (short exposure) strategies
- **Trade Splitting**: Splits trades that exceed position sizes
- **Put Option Handling**: Correctly handles inverted put option logic
- **Multiple Format Support**: Supports BOD, Contract, and MS position file formats
- **Sequential Processing**: Updates positions after each trade for accurate strategy assignment
- **GUI and CLI**: Both Streamlit web interface and command-line interface

## 📋 Strategy Rules

- **FULO (Futures/Options Long)**: 
  - Long Futures
  - Long Calls
  - Short Puts (short put = long exposure)

- **FUSH (Futures/Options Short)**:
  - Short Futures
  - Short Calls  
  - Long Puts (long put = short exposure)

## 🏗️ System Architecture

```
trade_strategy_system/
├── input_parser.py         # Parses position files (BOD/Contract/MS)
├── Trade_Parser.py         # Parses trade files (MS format)
├── position_manager.py     # Manages position state and updates
├── trade_processor.py      # Core trade processing logic
├── output_generator.py     # Generates output files
├── main.py                # Command-line interface
├── streamlit_app.py       # Web GUI interface
└── futures mapping.csv    # Symbol to Bloomberg ticker mapping
```

## 📦 Installation

1. Clone or download all the Python files to a directory

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Option 1: Streamlit GUI (Recommended)

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Upload your files in the sidebar:
   - Position file (initial positions)
   - Trade file (trades to process)
   - Mapping file (or use default)

3. Click "Process Trades" to run the analysis

4. View and download results from the interface

### Option 2: Command Line

```bash
python main.py position_file.xlsx trade_file.csv --mapping "futures mapping.csv" --output-prefix results
```

Arguments:
- `position_file`: Path to position file (BOD/Contract/MS format)
- `trade_file`: Path to trade file (MS format)
- `--mapping`: Path to symbol mapping file (default: futures mapping.csv)
- `--output-prefix`: Prefix for output files (default: output)
- `--verbose`: Enable detailed logging

## 📄 Input File Formats

### Position Files (3 formats supported)

1. **BOD Format**: 16+ columns with positions in columns 13-14
2. **Contract Format**: Contract IDs in column 3 with position in column 10
3. **MS Format**: Contract IDs in column 0 with positions in columns 19-20

### Trade File (MS Format)

14 columns:
- CP Code, TM Code, Scheme, TM Name, Instr, Symbol, Expiry Dt
- Lot Size, Strike Price, Option Type, B/S, Qty, Lots Traded, Avg Price

### Mapping File (CSV)

Columns:
1. Symbol
2. Bloomberg Ticker
3. Underlying (optional)
4. Reserved
5. Lot Size

## 📊 Output Files

The system generates 4 output files:

1. **Parsed Trades**: Original trades from parser
2. **Starting Positions**: Initial positions (Ticker, QTY)
3. **Processed Trades**: Main output with:
   - Original 14 trade columns
   - Strategy (FULO/FUSH)
   - Split? (Yes/No)
   - Opposite? (Yes/No)
   - Bloomberg_Ticker
4. **Final Positions**: Final positions after all trades (Ticker, QTY)

## 🔄 Processing Logic

1. **Load initial positions** from position file
2. **Process trades sequentially** in order
3. For each trade:
   - Check current position (updates after each trade)
   - If trade opposes position and exceeds it → **Split trade**
   - Assign strategy based on position direction
   - Update position for next trade
4. **Generate output files** with results

## 💡 Examples

### Example 1: Simple Trade
- Position: Long 100 Futures (FULO)
- Trade: Sell 80 Futures
- Output: -80, FULO, Split=No, Opposite=Yes

### Example 2: Split Trade
- Position: Long 100 Futures (FULO)
- Trade: Sell 120 Futures
- Output 1: -100, FULO, Split=Yes, Opposite=Yes
- Output 2: -20, FUSH, Split=Yes, Opposite=No

### Example 3: Put Option
- Position: Long 100 Puts (FUSH - short exposure)
- Trade: Buy 50 Puts
- Output: +50, FUSH, Split=No, Opposite=No

## ⚠️ Important Notes

- **Sequential Processing**: Trades must be processed in order as positions update after each trade
- **Put Options**: Remember puts have inverted directional logic
- **Split Trades**: Both split rows get "Yes" in Split? column
- **Opposite Flag**: "Yes" when strategy direction opposes trade direction

## 🐛 Troubleshooting

1. **No positions/trades found**: Check file format matches expected structure
2. **Unmapped symbols**: Add missing symbols to mapping file
3. **Excel password issues**: System tries common passwords, will prompt if needed
4. **Module import errors**: Ensure all files are in same directory

## 📞 Support

For issues or questions about the system:
1. Check input file formats match specifications
2. Verify mapping file has all required symbols
3. Review the summary report for processing details
4. Enable verbose logging for detailed debugging
