import os
from binance.client import Client
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

api_key = 'LOtOYSRqlH3lnIfxQSGldXsgMJMTK6VUFxh9tPMnWAQ71OYX5cLZXidCgRIU6RVQ'
api_secret = 'QTINJGoWZO8VEUQ1F5K0afngYDqArWyuU2w3ur4jsVhBmGr5yAF93xcHtAc43bcl'

client = Client(api_key, api_secret, testnet=True)

def get_available_symbols():
    tickers = client.get_all_tickers()
    symbols = [ticker['symbol'] for ticker in tickers if ticker['symbol'].endswith('USDT')]
    return symbols

def calculate_price_change(df):
    df['Price Change'] = df['Close'] - df['Open']
    df['Price Change (%)'] = df['Price Change'] / df['Open'] * 100
    df['Color'] = df['Price Change'].apply(lambda x: 'green' if x >= 0 else 'red')
    return df

def plot_price_change_chart(df, title):
    fig = go.Figure()

    # Add trace for price change percentage
    fig.add_trace(go.Bar(
        x=df['Time'],
        y=df['Price Change (%)'],
        marker_color=df['Color'],
        name='Price Change (%)'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price Change (%)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True,
        height=600,  # Increase height for better visibility
        width=1200,  # Increase width for better visibility
        margin=go.layout.Margin(
            l=100,  # left margin, in px
            r=100,  # right margin, in px
            t=100,  # top margin, in px
            b=100   # bottom margin, in px
        )
    )
    
    # Zoom in effect: Set x-axis to show the last N data points (e.g., last 20 data points)
    if len(df) > 20:
        fig.update_xaxes(range=[df['Time'].iloc[-21], df['Time'].iloc[-1]])

    # Zoom in effect: Adjust y-axis to a range around the current price changes
    fig.update_yaxes(
        range=[
            df['Price Change (%)'].min() - df['Price Change (%)'].std(),
            df['Price Change (%)'].max() + df['Price Change (%)'].std()
        ]
    )
    
    return fig

def plot_comparison_chart(avg_changes, title):
    fig = go.Figure()

    # Add trace for average percentage change comparison
    for symbol in avg_changes['Symbol'].unique():
        symbol_data = avg_changes[avg_changes['Symbol'] == symbol]
        fig.add_trace(go.Bar(
            x=symbol_data['Interval'],
            y=symbol_data['Average Change (%)'],
            name=symbol
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Interval',
        yaxis_title='Average Price Change (%)',
        barmode='group',
        xaxis_tickangle=-45,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,  # left margin, in px
            r=100,  # right margin, in px
            t=100,  # top margin, in px
            b=100   # bottom margin, in px
        )
    )
    
    return fig

def plot_overall_average_chart(avg_changes, title):
    fig = go.Figure()

    # Calculate overall average change percentage
    avg_overall_change = avg_changes['Average Change (%)'].mean()
    
    # Add trace for overall average change
    fig.add_trace(go.Bar(
        x=['All Selected Coins'],
        y=[avg_overall_change],
        marker_color='purple',
        name='Overall Average Change (%)'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title='Average Price Change (%)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,  # left margin, in px
            r=100,  # right margin, in px
            t=100,  # top margin, in px
            b=100   # bottom margin, in px
        )
    )
    
    return fig

def plot_direction_comparison_chart(direction_comparison, title):
    fig = go.Figure()

    # Add traces for direction comparison
    fig.add_trace(go.Bar(
        x=['Direction'],
        y=[direction_comparison['Same Direction']],
        marker_color='green',
        name='Same Direction'
    ))

    fig.add_trace(go.Bar(
        x=['Direction'],
        y=[-direction_comparison['Opposite Direction']],  # Negative for downward bar
        marker_color='red',
        name='Opposite Direction'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Direction',
        yaxis_title='Percentage (%)',
        xaxis_rangeslider_visible=False,
        barmode='relative',
        template='plotly_dark',
        autosize=True,
        height=600,
        width=600,  # Adjust width to be equal or less than height for better centering
        margin=go.layout.Margin(
            l=100,  # left margin, in px
            r=100,  # right margin, in px
            t=100,  # top margin, in px
            b=100   # bottom margin, in px
        )
    )

    # Adjust text labels and positioning
    fig.update_traces(
        texttemplate='%{y:.2f}%',  # Format text as percentage with two decimal places
        textposition='inside',     # Position text inside the bars
        textfont=dict(size=20),    # Adjust font size for better visibility
        textangle=0                # Ensure horizontal text alignment
    )

    # Ensure the y-axis range is appropriate for better centering
    fig.update_yaxes(
        range=[
            min(direction_comparison['Same Direction'], -direction_comparison['Opposite Direction']) - 10,
            max(direction_comparison['Same Direction'], -direction_comparison['Opposite Direction']) + 10
        ]
    )

    return fig

def plot_symbol_comparison_chart(symbol1, symbol2, intervals, title):
    fig = go.Figure()

    colors = {symbol1: 'green', symbol2: 'red'}
    
    for interval in intervals:
        # Fetch data for both symbols
        for symbol in [symbol1, symbol2]:
            candles = client.get_klines(symbol=symbol, interval=interval)
            data = []
            for candle in candles:
                open_time = datetime.datetime.fromtimestamp(candle[0] / 1000)
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                data.append([open_time, open_price, high_price, low_price, close_price])
            
            df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close'])
            df = calculate_price_change(df)
            
            avg_change = df['Price Change (%)'].mean()
            color = colors[symbol]
            
            # Add bars for each symbol
            fig.add_trace(go.Bar(
                x=[f'{interval} - {symbol}'],
                y=[avg_change],
                name=f'{symbol}',
                marker_color=color
            ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Interval',
        yaxis_title='Average Price Change (%)',
        barmode='group',
        xaxis_tickangle=-45,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,  # left margin, in px
            r=100,  # right margin, in px
            t=100,  # top margin, in px
            b=100   # bottom margin, in px
        )
    )

    return fig

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Compare 20 Coins", "Compare BTCUSDT and BCHUSDT"])

    if selection == "Compare 20 Coins":
        st.title("Cryptocurrency Price Analyze")

        symbols = get_available_symbols()
        
        # Input to select up to 20 symbols
        selected_symbols = st.multiselect(
            "Select up to 20 symbols", 
            options=symbols,
            max_selections=20
        )
        
        if selected_symbols:
            intervals = ['5m', '15m', '30m', '1h']
            avg_changes = []
            
            for symbol in selected_symbols:
                for interval in intervals:
                    candles = client.get_klines(symbol=symbol, interval=interval)
                    
                    data = []
                    for candle in candles:
                        open_time = datetime.datetime.fromtimestamp(candle[0] / 1000)
                        open_price = float(candle[1])
                        high_price = float(candle[2])
                        low_price = float(candle[3])
                        close_price = float(candle[4])
                        data.append([open_time, open_price, high_price, low_price, close_price])
                    
                    df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close'])
                    
                    # Calculate price change and assign colors
                    df = calculate_price_change(df)
                    
                    # Calculate average percentage change for each interval
                    avg_change = df['Price Change (%)'].mean()
                    avg_changes.append({'Symbol': symbol, 'Interval': interval, 'Average Change (%)': avg_change})

            # Create DataFrame for aggregated average percentage change
            avg_changes_df = pd.DataFrame(avg_changes)

            if len(selected_symbols) > 1:
                # Plot average price change by interval and symbol
                fig_comparison = plot_comparison_chart(avg_changes_df, "Average Price Change (%) by Interval and Symbol")
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Plot overall average price change for all selected coins
                fig_overall_avg = plot_overall_average_chart(avg_changes_df, "Overall Average Price Change (%) for Selected Coins")
                st.plotly_chart(fig_overall_avg, use_container_width=True)
                
                # Calculate direction comparison
                direction_comparison = {
                    'Same Direction': avg_changes_df[avg_changes_df['Average Change (%)'] >= 0].shape[0] / avg_changes_df.shape[0] * 100,
                    'Opposite Direction': avg_changes_df[avg_changes_df['Average Change (%)'] < 0].shape[0] / avg_changes_df.shape[0] * 100
                }
                
                # Plot direction comparison
                fig_direction_comparison = plot_direction_comparison_chart(direction_comparison, "Direction Comparison (%) of Selected Coins")
                st.plotly_chart(fig_direction_comparison, use_container_width=True)

            for symbol in selected_symbols:
                for interval in intervals:
                    st.write(f"### Price Change (%) for {symbol} ({interval})")
                    
                    candles = client.get_klines(symbol=symbol, interval=interval)
                    
                    data = []
                    for candle in candles:
                        open_time = datetime.datetime.fromtimestamp(candle[0] / 1000)
                        open_price = float(candle[1])
                        high_price = float(candle[2])
                        low_price = float(candle[3])
                        close_price = float(candle[4])
                        data.append([open_time, open_price, high_price, low_price, close_price])
                    
                    df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close'])
                    
                    # Calculate price change and assign colors
                    df = calculate_price_change(df)
                    
                    # Plot price change chart
                    fig_price_change = plot_price_change_chart(df, f"Price Change (%) ({symbol}, {interval})")
                    st.plotly_chart(fig_price_change, use_container_width=True)

    elif selection == "Compare BTCUSDT and BCHUSDT":
        st.title("Compare BTCUSDT and BCHUSDT")

        intervals = ['5m', '15m', '30m', '1h']
        fig_comparison = plot_symbol_comparison_chart('BTCUSDT', 'BCHUSDT', intervals, "BTCUSDT vs BCHUSDT Price Change (%)")
        st.plotly_chart(fig_comparison, use_container_width=True)

if __name__ == "__main__":
    main()
