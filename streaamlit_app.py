"""
Streamlit Application for Trade Strategy Processing System
Complete version with output display and downloads
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import logging
from datetime import datetime
import traceback
import io
import os

# Import our modules
from input_parser import InputParser
from Trade_Parser import TradeParser
from position_manager import PositionManager
from trade_processor import TradeProcessor
from output_generator import OutputGenerator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trade Strategy Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'output_files' not in st.session_state:
    st.session_state.output_files = {}
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

def main():
    st.title("🎯 Trade Strategy Processing System")
    st.markdown("### Assign strategies to trades based on position directions")
    
    # Sidebar for file inputs
    with st.sidebar:
        st.header("📁 Input Files")
        
        # Position file
        st.subheader("1. Position File")
        position_file = st.file_uploader(
            "Upload Position File (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            key='position_file',
            help="BOD, Contract, or MS format position file"
        )
        
        # Trade file
        st.subheader("2. Trade File")
        trade_file = st.file_uploader(
            "Upload Trade File (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            key='trade_file',
            help="MS format trade file"
        )
        
        # Mapping file
        st.subheader("3. Mapping File")
        
        # Check for default mapping files
        default_exists = False
        default_path = None
        for possible_name in ["futures mapping.csv", "futures_mapping.csv"]:
            if Path(possible_name).exists():
                default_exists = True
                default_path = possible_name
                break
        
        if default_exists:
            use_default = st.radio(
                "Mapping file source:",
                ["Use default from repository", "Upload custom mapping file"],
                index=0
            )
            
            if use_default == "Upload custom mapping file":
                mapping_file = st.file_uploader(
                    "Upload Symbol Mapping File (CSV)",
                    type=['csv'],
                    key='mapping_file'
                )
            else:
                mapping_file = None
                st.success(f"✓ Using {default_path}")
        else:
            st.warning("No default mapping file found. Please upload one.")
            mapping_file = st.file_uploader(
                "Upload Symbol Mapping File (CSV)",
                type=['csv'],
                key='mapping_file',
                help="Required: Symbol to Bloomberg ticker mapping"
            )
            use_default = None
        
        st.divider()
        
        # Process button
        can_process = (
            position_file is not None and 
            trade_file is not None and 
            (mapping_file is not None or (use_default == "Use default from repository" and default_exists))
        )
        
        process_button = st.button(
            "🚀 Process Trades",
            type="primary",
            use_container_width=True,
            disabled=not can_process
        )
        
        # Reset button if already processed
        if st.session_state.processed:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                st.session_state.processed = False
                st.session_state.output_files = {}
                st.session_state.dataframes = {}
                st.rerun()
    
    # Main content area
    if not st.session_state.processed:
        # Instructions
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **How it works:**
            1. Upload your position file (initial positions)
            2. Upload your trade file (trades to process)
            3. Use default or upload mapping file
            4. Click 'Process Trades' to run
            """)
        
        with col2:
            st.success("""
            **Strategy Rules:**
            - **FULO**: Long Futures/Calls, Short Puts
            - **FUSH**: Short Futures/Calls, Long Puts
            - Trades inherit position's strategy when closing
            - Trades split when exceeding position size
            """)
        
        # Process when button clicked
        if process_button:
            process_files(
                position_file, 
                trade_file, 
                mapping_file, 
                use_default, 
                default_path if default_exists else None
            )
    
    else:
        # Display results
        display_results()

def process_files(position_file, trade_file, mapping_file, use_default, default_path):
    """Process the uploaded files"""
    try:
        with st.spinner("Processing files..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save uploaded files
            status_text.text("Saving uploaded files...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(position_file.name).suffix) as tmp_pos:
                tmp_pos.write(position_file.getbuffer())
                pos_file_path = tmp_pos.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(trade_file.name).suffix) as tmp_trade:
                tmp_trade.write(trade_file.getbuffer())
                trade_file_path = tmp_trade.name
            
            # Handle mapping file
            if mapping_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_map:
                    tmp_map.write(mapping_file.getbuffer())
                    mapping_file_path = tmp_map.name
            elif use_default == "Use default from repository":
                mapping_file_path = default_path
            else:
                st.error("Please provide a mapping file")
                return
            
            # Initialize parsers
            status_text.text("Initializing parsers...")
            progress_bar.progress(20)
            
            input_parser = InputParser(mapping_file_path)
            trade_parser = TradeParser(mapping_file_path)
            
            # Parse position file
            status_text.text("Parsing position file...")
            progress_bar.progress(30)
            
            positions = input_parser.parse_file(pos_file_path)
            if not positions:
                st.error("❌ No positions found in the position file")
                return
            
            st.success(f"✅ Parsed {len(positions)} positions from {input_parser.format_type} format")
            
            # Show unmapped symbols
            if input_parser.unmapped_symbols:
                with st.expander(f"⚠️ {len(input_parser.unmapped_symbols)} unmapped position symbols"):
                    st.dataframe(pd.DataFrame(input_parser.unmapped_symbols))
            
            # Parse trade file
            status_text.text("Parsing trade file...")
            progress_bar.progress(40)
            
            # Read raw trade DataFrame
            try:
                if trade_file_path.endswith('.csv'):
                    trade_df = pd.read_csv(trade_file_path, header=None)
                else:
                    trade_df = pd.read_excel(trade_file_path, header=None)
            except:
                if trade_file_path.endswith('.csv'):
                    trade_df = pd.read_csv(trade_file_path)
                else:
                    trade_df = pd.read_excel(trade_file_path)
            
            trades = trade_parser.parse_trade_file(trade_file_path)
            if not trades:
                st.error("❌ No trades found in the trade file")
                return
            
            st.success(f"✅ Parsed {len(trades)} trades from {trade_parser.format_type} format")
            
            # Show unmapped symbols
            if trade_parser.unmapped_symbols:
                with st.expander(f"⚠️ {len(trade_parser.unmapped_symbols)} unmapped trade symbols"):
                    st.dataframe(pd.DataFrame(trade_parser.unmapped_symbols))
            
            # DIAGNOSTIC SECTION
            with st.expander("🔍 Ticker Matching Diagnostic"):
                show_ticker_diagnostic(positions, trades)
            
            # Initialize position manager
            status_text.text("Initializing position manager...")
            progress_bar.progress(50)
            
            position_manager = PositionManager()
            starting_positions_df = position_manager.initialize_from_positions(positions)
            
            # Process trades
            status_text.text("Processing trades against positions...")
            progress_bar.progress(60)
            
            trade_processor = TradeProcessor(position_manager)
            output_gen = OutputGenerator()
            
            # Create parsed trades DataFrame
            parsed_trades_df = output_gen.create_trade_dataframe_from_positions(trades)
            
            # Process trades
            processed_trades_df = trade_processor.process_trades(trades, trade_df)
            
            # Get final positions
            status_text.text("Calculating final positions...")
            progress_bar.progress(80)
            
            final_positions_df = position_manager.get_final_positions()
            
            # Generate output files
            status_text.text("Generating output files...")
            progress_bar.progress(90)
            
            output_files = output_gen.save_all_outputs(
                parsed_trades_df,
                starting_positions_df,
                processed_trades_df,
                final_positions_df,
                file_prefix="trade_strategy"
            )
            
            # Store results in session state
            st.session_state.output_files = output_files
            st.session_state.dataframes = {
                'parsed_trades': parsed_trades_df,
                'starting_positions': starting_positions_df,
                'processed_trades': processed_trades_df,
                'final_positions': final_positions_df
            }
            st.session_state.processed = True
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Success message
            st.balloons()
            st.success("🎉 Processing completed successfully!")
            
            # Show summary
            show_processing_summary(starting_positions_df, trades, processed_trades_df, final_positions_df)
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.code(traceback.format_exc())

def show_ticker_diagnostic(positions, trades):
    """Show diagnostic information about ticker matching"""
    st.subheader("Ticker Matching Analysis")
    
    # Get position tickers
    position_details = []
    for pos in positions:
        position_details.append({
            'Symbol': pos.symbol,
            'Bloomberg Ticker': pos.bloomberg_ticker,
            'Quantity': pos.position_lots,
            'Type': pos.security_type
        })
    
    # Get trade tickers
    trade_details = []
    for trade in trades:
        trade_details.append({
            'Symbol': trade.symbol,
            'Bloomberg Ticker': trade.bloomberg_ticker,
            'Quantity': trade.position_lots,
            'Type': trade.security_type
        })
    
    # Create sets for comparison
    position_tickers = {p['Bloomberg Ticker'] for p in position_details}
    trade_tickers = {t['Bloomberg Ticker'] for t in trade_details}
    
    # Find matches
    matching = position_tickers.intersection(trade_tickers)
    unmatched_pos = position_tickers - trade_tickers
    unmatched_trades = trade_tickers - position_tickers
    
    # Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matching Tickers", len(matching))
    with col2:
        st.metric("Unmatched Positions", len(unmatched_pos))
    with col3:
        st.metric("Unmatched Trades", len(unmatched_trades))
    
    # Show expected splits
    split_count = 0
    for ticker in matching:
        pos = next(p for p in position_details if p['Bloomberg Ticker'] == ticker)
        trade = next(t for t in trade_details if t['Bloomberg Ticker'] == ticker)
        
        if (pos['Quantity'] > 0 and trade['Quantity'] < 0) or \
           (pos['Quantity'] < 0 and trade['Quantity'] > 0):
            if abs(trade['Quantity']) > abs(pos['Quantity']):
                split_count += 1
    
    st.info(f"Expected splits: {split_count}")

def show_processing_summary(starting_df, trades, processed_df, final_df):
    """Show summary statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Starting Positions", len(starting_df))
    
    with col2:
        st.metric("Trades Processed", len(trades))
    
    with col3:
        split_count = 0
        if 'Split?' in processed_df.columns:
            split_count = len(processed_df[processed_df['Split?'] == 'Yes'])
        st.metric("Split Trades", split_count)
    
    with col4:
        st.metric("Final Positions", len(final_df))

def display_results():
    """Display the processing results"""
    st.header("📊 Processing Results")
    
    # Create tabs
    tabs = st.tabs([
        "📈 Processed Trades",
        "📍 Starting Positions",
        "📍 Final Positions",
        "📋 Parsed Trades",
        "📥 Download Files"
    ])
    
    # Tab 1: Processed Trades
    with tabs[0]:
        if 'processed_trades' in st.session_state.dataframes:
            df = st.session_state.dataframes['processed_trades']
            
            st.subheader("Processed Trades with Strategy Assignment")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                if 'Strategy' in df.columns:
                    st.metric("FULO", len(df[df['Strategy'] == 'FULO']))
            with col3:
                if 'Strategy' in df.columns:
                    st.metric("FUSH", len(df[df['Strategy'] == 'FUSH']))
            with col4:
                if 'Opposite?' in df.columns:
                    st.metric("Opposite", len(df[df['Opposite?'] == 'Yes']))
            
            # Show dataframe
            st.dataframe(df, use_container_width=True, height=400)
            
            # Show splits
            if 'Split?' in df.columns:
                splits = df[df['Split?'] == 'Yes']
                if not splits.empty:
                    st.subheader("🔀 Split Trades")
                    st.dataframe(splits, use_container_width=True)
        else:
            st.warning("No processed trades data available")
    
    # Tab 2: Starting Positions
    with tabs[1]:
        if 'starting_positions' in st.session_state.dataframes:
            df = st.session_state.dataframes['starting_positions']
            
            st.subheader("Starting Positions")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", len(df))
            with col2:
                long = len(df[df['QTY'] > 0]) if len(df) > 0 else 0
                short = len(df[df['QTY'] < 0]) if len(df) > 0 else 0
                st.metric("Long/Short", f"{long}/{short}")
            
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No starting positions data available")
    
    # Tab 3: Final Positions
    with tabs[2]:
        if 'final_positions' in st.session_state.dataframes:
            df = st.session_state.dataframes['final_positions']
            
            st.subheader("Final Positions")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", len(df))
            with col2:
                long = len(df[df['QTY'] > 0]) if len(df) > 0 else 0
                short = len(df[df['QTY'] < 0]) if len(df) > 0 else 0
                st.metric("Long/Short", f"{long}/{short}")
            
            st.dataframe(df, use_container_width=True)
            
            # Position changes
            if 'starting_positions' in st.session_state.dataframes:
                start_df = st.session_state.dataframes['starting_positions']
                st.subheader("📊 Position Changes")
                
                start_tickers = set(start_df['Ticker'].unique())
                final_tickers = set(df['Ticker'].unique()) if len(df) > 0 else set()
                
                new_pos = final_tickers - start_tickers
                closed_pos = start_tickers - final_tickers
                
                col1, col2 = st.columns(2)
                with col1:
                    if new_pos:
                        st.info(f"**{len(new_pos)} New Positions**")
                with col2:
                    if closed_pos:
                        st.warning(f"**{len(closed_pos)} Closed Positions**")
        else:
            st.warning("No final positions data available")
    
    # Tab 4: Parsed Trades
    with tabs[3]:
        if 'parsed_trades' in st.session_state.dataframes:
            df = st.session_state.dataframes['parsed_trades']
            
            st.subheader("Parsed Trades (Raw)")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No parsed trades data available")
    
    # Tab 5: Download Files
    with tabs[4]:
        st.subheader("📥 Download Output Files")
        
        if st.session_state.output_files:
            st.success("✅ All output files are ready for download!")
            
            # Create columns for download buttons
            col1, col2 = st.columns(2)
            
            file_order = [
                ('parsed_trades', '1️⃣ Parsed Trades (CSV)', 'text/csv'),
                ('starting_positions', '2️⃣ Starting Positions (CSV)', 'text/csv'),
                ('processed_trades', '3️⃣ Processed Trades (CSV)', 'text/csv'),
                ('processed_trades_excel', '📊 Processed Trades (Excel)', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                ('final_positions', '4️⃣ Final Positions (CSV)', 'text/csv'),
                ('summary', '📝 Summary Report (TXT)', 'text/plain')
            ]
            
            for idx, (file_key, label, mime) in enumerate(file_order):
                if file_key in st.session_state.output_files:
                    file_path = st.session_state.output_files[file_key]
                    if file_path and Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Use columns for layout
                        target_col = col1 if idx % 2 == 0 else col2
                        with target_col:
                            st.download_button(
                                label=label,
                                data=file_data,
                                file_name=Path(file_path).name,
                                mime=mime,
                                key=f"download_{file_key}",
                                use_container_width=True
                            )
            
            # Show summary report content
            st.divider()
            if 'summary' in st.session_state.output_files:
                summary_path = st.session_state.output_files['summary']
                if Path(summary_path).exists():
                    with st.expander("📝 View Summary Report"):
                        with open(summary_path, 'r') as f:
                            st.text(f.read())
        else:
            st.error("❌ No output files available. Please process files first.")

if __name__ == "__main__":
    main()
