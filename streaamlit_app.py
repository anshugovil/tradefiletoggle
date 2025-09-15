"""
Streamlit Application - ENHANCED VERSION
Processes and displays results with missing mappings tracking
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import logging
from datetime import datetime
import traceback
import os

# Import modules
from input_parser import InputParser
from Trade_Parser import TradeParser
from position_manager import PositionManager
from trade_processor import TradeProcessor
from output_generator import OutputGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Trade Strategy Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; }
    .stDownloadButton button { 
        width: 100%; 
        background-color: #4CAF50; 
        color: white; 
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🎯 Trade Strategy Processing System")
    st.markdown("### Assign strategies to trades based on position directions")
    
    # Initialize session state if needed
    if 'output_files' not in st.session_state:
        st.session_state.output_files = {}
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}
    if 'missing_mappings' not in st.session_state:
        st.session_state.missing_mappings = {'positions': [], 'trades': []}
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Input Files")
        
        position_file = st.file_uploader(
            "1. Position File",
            type=['xlsx', 'xls', 'csv'],
            key='position_file',
            help="BOD, Contract, or MS format"
        )
        
        trade_file = st.file_uploader(
            "2. Trade File",
            type=['xlsx', 'xls', 'csv'],
            key='trade_file',
            help="MS format trade file"
        )
        
        st.subheader("3. Mapping File")
        
        # Check for default mapping
        default_path = None
        for name in ["futures mapping.csv", "futures_mapping.csv"]:
            if Path(name).exists():
                default_path = name
                break
        
        if default_path:
            use_default = st.radio(
                "Mapping source:",
                ["Use default from repository", "Upload custom"],
                index=0
            )
            
            if use_default == "Upload custom":
                mapping_file = st.file_uploader(
                    "Upload Mapping File",
                    type=['csv'],
                    key='mapping_file'
                )
            else:
                mapping_file = None
                st.success(f"✓ Using {default_path}")
        else:
            st.warning("Upload mapping file (required)")
            mapping_file = st.file_uploader(
                "Upload Mapping File",
                type=['csv'],
                key='mapping_file'
            )
            use_default = None
        
        st.divider()
        
        # Process button
        can_process = (
            position_file is not None and 
            trade_file is not None and 
            (mapping_file is not None or (use_default == "Use default from repository" and default_path))
        )
        
        if st.button("🚀 Process Trades", type="primary", use_container_width=True, disabled=not can_process):
            # Process immediately when button is clicked
            process_and_display(position_file, trade_file, mapping_file, use_default, default_path)
    
    # Main content area
    if st.session_state.dataframes:
        # Display results if we have them
        display_results()
    else:
        # Show instructions
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Steps:**
            1. Upload position file
            2. Upload trade file
            3. Select/upload mapping file
            4. Click 'Process Trades'
            """)
        
        with col2:
            st.success("""
            **Strategy Rules:**
            - **FULO**: Long Futures/Calls, Short Puts
            - **FUSH**: Short Futures/Calls, Long Puts
            - Closing trades inherit position's strategy
            """)

def process_and_display(position_file, trade_file, mapping_file, use_default, default_path):
    """Process files and immediately display results"""
    try:
        # Clear previous results
        st.session_state.output_files = {}
        st.session_state.dataframes = {}
        st.session_state.missing_mappings = {'positions': [], 'trades': []}
        
        with st.spinner("Processing..."):
            # Save uploaded files
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(position_file.name).suffix) as tmp:
                tmp.write(position_file.getbuffer())
                pos_path = tmp.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(trade_file.name).suffix) as tmp:
                tmp.write(trade_file.getbuffer())
                trade_path = tmp.name
            
            if mapping_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(mapping_file.getbuffer())
                    map_path = tmp.name
            else:
                map_path = default_path
            
            # Parse positions
            input_parser = InputParser(map_path)
            positions = input_parser.parse_file(pos_path)
            
            if not positions:
                st.error("❌ No positions found")
                return
            
            st.success(f"✅ Parsed {len(positions)} positions ({input_parser.format_type} format)")
            
            # Parse trades
            trade_parser = TradeParser(map_path)
            
            if trade_path.endswith('.csv'):
                trade_df = pd.read_csv(trade_path, header=None)
            else:
                trade_df = pd.read_excel(trade_path, header=None)
            
            trades = trade_parser.parse_trade_file(trade_path)
            
            if not trades:
                st.error("❌ No trades found")
                return
            
            st.success(f"✅ Parsed {len(trades)} trades ({trade_parser.format_type} format)")
            
            # Check for missing mappings
            missing_positions = len(input_parser.unmapped_symbols) if hasattr(input_parser, 'unmapped_symbols') else 0
            missing_trades = len(trade_parser.unmapped_symbols) if hasattr(trade_parser, 'unmapped_symbols') else 0
            
            if missing_positions > 0 or missing_trades > 0:
                st.warning(f"⚠️ Found unmapped symbols: {missing_positions} from positions, {missing_trades} from trades")
                st.session_state.missing_mappings['positions'] = input_parser.unmapped_symbols if missing_positions > 0 else []
                st.session_state.missing_mappings['trades'] = trade_parser.unmapped_symbols if missing_trades > 0 else []
            
            # Show diagnostic
            with st.expander("🔍 Ticker Matching & Missing Mappings"):
                show_diagnostic(positions, trades, input_parser, trade_parser)
            
            # Process trades
            position_manager = PositionManager()
            starting_positions_df = position_manager.initialize_from_positions(positions)
            
            trade_processor = TradeProcessor(position_manager)
            output_gen = OutputGenerator()
            
            parsed_trades_df = output_gen.create_trade_dataframe_from_positions(trades)
            processed_trades_df = trade_processor.process_trades(trades, trade_df)
            final_positions_df = position_manager.get_final_positions()
            
            # Generate output files with parsers for missing mappings
            output_files = output_gen.save_all_outputs(
                parsed_trades_df,
                starting_positions_df,
                processed_trades_df,
                final_positions_df,
                file_prefix="output",
                input_parser=input_parser,
                trade_parser=trade_parser
            )
            
            # Store in session state
            st.session_state.output_files = output_files
            st.session_state.dataframes = {
                'parsed_trades': parsed_trades_df,
                'starting_positions': starting_positions_df,
                'processed_trades': processed_trades_df,
                'final_positions': final_positions_df
            }
            
            # Show summary
            st.success("✅ Processing complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Starting Positions", len(starting_positions_df))
            with col2:
                st.metric("Trades", len(trades))
            with col3:
                splits = len(processed_trades_df[processed_trades_df['Split?'] == 'Yes']) if 'Split?' in processed_trades_df.columns else 0
                st.metric("Splits", splits)
            with col4:
                st.metric("Final Positions", len(final_positions_df))
            
            # Force a rerun to display results
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.code(traceback.format_exc())

def show_diagnostic(positions, trades, input_parser=None, trade_parser=None):
    """Show ticker matching diagnostic and missing mappings"""
    pos_tickers = {p.bloomberg_ticker for p in positions}
    trade_tickers = {t.bloomberg_ticker for t in trades}
    
    matching = pos_tickers.intersection(trade_tickers)
    unmatched_pos = pos_tickers - trade_tickers
    unmatched_trades = trade_tickers - pos_tickers
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matching", len(matching))
    with col2:
        st.metric("Unmatched Positions", len(unmatched_pos))
    with col3:
        st.metric("Unmatched Trades", len(unmatched_trades))
    
    # Show missing mappings if any
    if input_parser and hasattr(input_parser, 'unmapped_symbols') and input_parser.unmapped_symbols:
        st.markdown("### ⚠️ Missing Mappings from Position File")
        missing_df = pd.DataFrame(input_parser.unmapped_symbols)
        st.dataframe(missing_df, use_container_width=True, height=150)
    
    if trade_parser and hasattr(trade_parser, 'unmapped_symbols') and trade_parser.unmapped_symbols:
        st.markdown("### ⚠️ Missing Mappings from Trade File")
        missing_df = pd.DataFrame(trade_parser.unmapped_symbols)
        st.dataframe(missing_df, use_container_width=True, height=150)

def display_results():
    """Display processing results"""
    st.header("📊 Results")
    
    # Show warning if there are missing mappings
    if (st.session_state.missing_mappings['positions'] or 
        st.session_state.missing_mappings['trades']):
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Warning:</strong> Some symbols could not be mapped. 
        Download the MISSING_MAPPINGS file to see which symbols need to be added to your mapping file.
        </div>
        """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📈 Processed Trades",
        "📍 Starting Positions",
        "📍 Final Positions",
        "📋 Parsed Trades",
        "📥 Downloads"
    ])
    
    with tabs[0]:
        if 'processed_trades' in st.session_state.dataframes:
            df = st.session_state.dataframes['processed_trades']
            st.subheader("Processed Trades")
            
            col1, col2, col3 = st.columns(3)
            if 'Strategy' in df.columns:
                with col1:
                    st.metric("FULO", len(df[df['Strategy'] == 'FULO']))
                with col2:
                    st.metric("FUSH", len(df[df['Strategy'] == 'FUSH']))
            if 'Opposite?' in df.columns:
                with col3:
                    st.metric("Opposite", len(df[df['Opposite?'] == 'Yes']))
            
            st.dataframe(df, use_container_width=True, height=400)
            
            if 'Split?' in df.columns:
                splits = df[df['Split?'] == 'Yes']
                if not splits.empty:
                    st.subheader("🔀 Split Trades")
                    st.dataframe(splits, use_container_width=True)
    
    with tabs[1]:
        if 'starting_positions' in st.session_state.dataframes:
            df = st.session_state.dataframes['starting_positions']
            st.subheader("Starting Positions")
            st.dataframe(df, use_container_width=True)
    
    with tabs[2]:
        if 'final_positions' in st.session_state.dataframes:
            df = st.session_state.dataframes['final_positions']
            st.subheader("Final Positions")
            st.dataframe(df, use_container_width=True)
    
    with tabs[3]:
        if 'parsed_trades' in st.session_state.dataframes:
            df = st.session_state.dataframes['parsed_trades']
            st.subheader("Parsed Trades")
            st.dataframe(df, use_container_width=True)
    
    with tabs[4]:
        st.subheader("📥 Download Files")
        
        # Highlight missing mappings file if it exists
        if 'missing_mappings' in st.session_state.output_files:
            st.markdown("### ⚠️ Missing Mappings Files")
            col1, col2 = st.columns(2)
            
            with col1:
                missing_path = st.session_state.output_files['missing_mappings']
                if missing_path and Path(missing_path).exists():
                    with open(missing_path, 'rb') as f:
                        data = f.read()
                    st.download_button(
                        "📝 Download Missing Mappings Report",
                        data,
                        file_name=Path(missing_path).name,
                        mime='text/csv',
                        key="dl_missing",
                        use_container_width=True,
                        help="List of all unmapped symbols"
                    )
            
            with col2:
                # Check for template file
                template_path = Path(missing_path).parent / f"MAPPING_TEMPLATE_{Path(missing_path).stem.split('_')[-1]}.csv"
                if template_path.exists():
                    with open(template_path, 'rb') as f:
                        data = f.read()
                    st.download_button(
                        "📄 Download Mapping Template",
                        data,
                        file_name=template_path.name,
                        mime='text/csv',
                        key="dl_template",
                        use_container_width=True,
                        help="Ready-to-fill template for your mapping file"
                    )
            
            st.markdown("---")
        
        st.markdown("### 📊 Output Files")
        
        if st.session_state.output_files:
            for key, path in st.session_state.output_files.items():
                if key in ['missing_mappings']:  # Already handled above
                    continue
                    
                if path and Path(path).exists():
                    with open(path, 'rb') as f:
                        data = f.read()
                    
                    mime = 'text/csv'
                    if 'excel' in key:
                        mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    elif 'summary' in key:
                        mime = 'text/plain'
                    
                    label = key.replace('_', ' ').title()
                    st.download_button(
                        f"📄 {label}",
                        data,
                        file_name=Path(path).name,
                        mime=mime,
                        key=f"dl_{key}",
                        use_container_width=True
                    )
        
        # Reset button
        if st.button("🔄 Process New Files", type="secondary", use_container_width=True):
            st.session_state.output_files = {}
            st.session_state.dataframes = {}
            st.session_state.missing_mappings = {'positions': [], 'trades': []}
            st.rerun()

if __name__ == "__main__":
    main()
