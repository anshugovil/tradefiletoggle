"""
Streamlit Application for Trade Strategy Processing System
Main GUI interface for processing trades against positions
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import logging
from datetime import datetime
import traceback
import io

# Import our modules - these should be in the same directory
from input_parser import InputParser
from Trade_Parser import TradeParser
from position_manager import PositionManager
from trade_processor import TradeProcessor
from output_generator import OutputGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trade Strategy Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    # Sidebar for file inputs and configuration
    with st.sidebar:
        st.header("📁 Input Files")
        
        # Position file upload
        st.subheader("1. Position File")
        position_file = st.file_uploader(
            "Upload Position File (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            key='position_file',
            help="BOD, Contract, or MS format position file"
        )
        
        # Trade file upload
        st.subheader("2. Trade File")
        trade_file = st.file_uploader(
            "Upload Trade File (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            key='trade_file',
            help="MS format trade file"
        )
        
        # Mapping file upload
        st.subheader("3. Mapping File")
        mapping_file = st.file_uploader(
            "Upload Symbol Mapping File (CSV)",
            type=['csv'],
            key='mapping_file',
            help="Symbol to Bloomberg ticker mapping"
        )
        
        # Use default mapping file option
        use_default_mapping = st.checkbox(
            "Use default mapping file",
            value=False,
            help="Use 'futures mapping.csv' if available locally"
        )
        
        st.divider()
        
        # Process button
        process_button = st.button(
            "🚀 Process Trades",
            type="primary",
            use_container_width=True,
            disabled=(position_file is None or trade_file is None or (mapping_file is None and not use_default_mapping))
        )
    
    # Main content area
    if not st.session_state.processed:
        # Instructions and information
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **How it works:**
            1. Upload your position file (initial positions)
            2. Upload your trade file (trades to process)
            3. Upload mapping file (required)
            4. Click 'Process Trades' to run the analysis
            """)
        
        with col2:
            st.success("""
            **Strategy Assignment Rules:**
            - **FULO**: Long Futures/Calls, Short Puts
            - **FUSH**: Short Futures/Calls, Long Puts
            - Trades inherit position's strategy when closing
            - Trades split when exceeding position size
            """)
        
        # Process files when button clicked
        if process_button:
            if position_file and trade_file:
                process_files(position_file, trade_file, mapping_file, use_default_mapping)
    
    else:
        # Display results
        display_results()

def process_files(position_file, trade_file, mapping_file, use_default_mapping):
    """Process the uploaded files"""
    try:
        with st.spinner("Processing files..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Save uploaded files temporarily
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
            elif use_default_mapping and Path("futures mapping.csv").exists():
                mapping_file_path = "futures mapping.csv"
            else:
                st.error("Please provide a mapping file")
                return
            
            # Step 2: Initialize parsers
            status_text.text("Initializing parsers...")
            progress_bar.progress(20)
            
            input_parser = InputParser(mapping_file_path)
            trade_parser = TradeParser(mapping_file_path)
            
            # Step 3: Parse position file
            status_text.text("Parsing position file...")
            progress_bar.progress(30)
            
            positions = input_parser.parse_file(pos_file_path)
            if not positions:
                st.error("No positions found in the position file")
                return
            
            st.success(f"✅ Parsed {len(positions)} positions from {input_parser.format_type} format")
            
            # Step 4: Parse trade file
            status_text.text("Parsing trade file...")
            progress_bar.progress(40)
            
            # Read the raw trade DataFrame
            if trade_file_path.endswith('.csv'):
                trade_df = pd.read_csv(trade_file_path)
            else:
                trade_df = pd.read_excel(trade_file_path)
            
            trades = trade_parser.parse_trade_file(trade_file_path)
            if not trades:
                st.error("No trades found in the trade file")
                return
            
            st.success(f"✅ Parsed {len(trades)} trades from {trade_parser.format_type} format")
            
            # Step 5: Initialize position manager
            status_text.text("Initializing position manager...")
            progress_bar.progress(50)
            
            position_manager = PositionManager()
            starting_positions_df = position_manager.initialize_from_positions(positions)
            
            # Step 6: Process trades
            status_text.text("Processing trades against positions...")
            progress_bar.progress(60)
            
            trade_processor = TradeProcessor(position_manager)
            
            # Create parsed trades DataFrame
            output_gen = OutputGenerator()
            parsed_trades_df = output_gen.create_trade_dataframe_from_positions(trades)
            
            # Process trades
            processed_trades_df = trade_processor.process_trades(trades, trade_df)
            
            # Step 7: Get final positions
            status_text.text("Calculating final positions...")
            progress_bar.progress(80)
            
            final_positions_df = position_manager.get_final_positions()
            
            # Step 8: Generate outputs
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
            
            # Display success message
            st.balloons()
            st.success("🎉 Processing completed successfully!")
            
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Starting Positions", len(starting_positions_df))
            with col2:
                st.metric("Trades Processed", len(trades))
            with col3:
                split_count = len(processed_trades_df[processed_trades_df['Split?'] == 'Yes']) if 'Split?' in processed_trades_df else 0
                st.metric("Split Trades", split_count)
            with col4:
                st.metric("Final Positions", len(final_positions_df))
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.code(traceback.format_exc())
        logger.error(f"Processing error: {e}", exc_info=True)

def display_results():
    """Display the processing results"""
    st.header("📊 Processing Results")
    
    # Add a reset button
    if st.button("🔄 Process New Files", type="secondary"):
        st.session_state.processed = False
        st.session_state.output_files = {}
        st.session_state.dataframes = {}
        st.rerun()
    
    # Create tabs for different outputs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Processed Trades",
        "📍 Starting Positions",
        "📍 Final Positions", 
        "📋 Parsed Trades",
        "📥 Download Files"
    ])
    
    with tab1:
        st.subheader("Processed Trades with Strategy Assignment")
        if 'processed_trades' in st.session_state.dataframes:
            df = st.session_state.dataframes['processed_trades']
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'Strategy' in df.columns:
                    st.metric("FULO Trades", len(df[df['Strategy'] == 'FULO']))
            with col2:
                if 'Strategy' in df.columns:
                    st.metric("FUSH Trades", len(df[df['Strategy'] == 'FUSH']))
            with col3:
                if 'Opposite?' in df.columns:
                    st.metric("Opposite Trades", len(df[df['Opposite?'] == 'Yes']))
            
            # Display the dataframe
            st.dataframe(df, use_container_width=True, height=400)
            
            # Highlight split trades
            if 'Split?' in df.columns:
                split_trades = df[df['Split?'] == 'Yes']
                if not split_trades.empty:
                    st.subheader("🔀 Split Trades")
                    st.dataframe(split_trades, use_container_width=True)
    
    with tab2:
        st.subheader("Starting Positions")
        if 'starting_positions' in st.session_state.dataframes:
            df = st.session_state.dataframes['starting_positions']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Positions", len(df))
            with col2:
                long_count = len(df[df['QTY'] > 0])
                short_count = len(df[df['QTY'] < 0])
                st.metric("Long/Short", f"{long_count}/{short_count}")
            
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.subheader("Final Positions After Trade Processing")
        if 'final_positions' in st.session_state.dataframes:
            df = st.session_state.dataframes['final_positions']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Positions", len(df))
            with col2:
                if len(df) > 0:
                    long_count = len(df[df['QTY'] > 0])
                    short_count = len(df[df['QTY'] < 0])
                    st.metric("Long/Short", f"{long_count}/{short_count}")
            
            st.dataframe(df, use_container_width=True)
            
            # Show position changes
            if 'starting_positions' in st.session_state.dataframes:
                start_df = st.session_state.dataframes['starting_positions']
                st.subheader("📊 Position Changes")
                
                start_tickers = set(start_df['Ticker'].unique())
                final_tickers = set(df['Ticker'].unique()) if len(df) > 0 else set()
                
                new_positions = final_tickers - start_tickers
                closed_positions = start_tickers - final_tickers
                
                col1, col2 = st.columns(2)
                with col1:
                    if new_positions:
                        st.info(f"**New Positions Opened ({len(new_positions)}):**")
                        for ticker in list(new_positions)[:10]:
                            st.write(f"• {ticker}")
                        if len(new_positions) > 10:
                            st.write(f"... and {len(new_positions) - 10} more")
                
                with col2:
                    if closed_positions:
                        st.warning(f"**Positions Closed ({len(closed_positions)}):**")
                        for ticker in list(closed_positions)[:10]:
                            st.write(f"• {ticker}")
                        if len(closed_positions) > 10:
                            st.write(f"... and {len(closed_positions) - 10} more")
    
    with tab4:
        st.subheader("Parsed Trades (Raw)")
        if 'parsed_trades' in st.session_state.dataframes:
            df = st.session_state.dataframes['parsed_trades']
            st.dataframe(df, use_container_width=True)
    
    with tab5:
        st.subheader("📥 Download Output Files")
        
        # Create download buttons for each file
        for file_type, file_path in st.session_state.output_files.items():
            if file_path and Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                file_name = Path(file_path).name
                if 'excel' in file_type:
                    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    label = f"📊 {file_name}"
                elif 'summary' in file_type:
                    mime_type = 'text/plain'
                    label = f"📄 {file_name}"
                else:
                    mime_type = 'text/csv'
                    label = f"📈 {file_name}"
                
                st.download_button(
                    label=label,
                    data=file_bytes,
                    file_name=file_name,
                    mime=mime_type,
                    key=f"download_{file_type}"
                )

if __name__ == "__main__":
    main()
