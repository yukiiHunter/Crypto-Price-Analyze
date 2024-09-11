import os
from binance.client import Client
import datetime
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import threading
import time
import pytz

api_key = 'LOtOYSRqlH3lnIfxQSGldXsgMJMTK6VUFxh9tPMnWAQ71OYX5cLZXidCgRIU6RVQ'
api_secret = 'QTINJGoWZO8VEUQ1F5K0afngYDqArWyuU2w3ur4jsVhBmGr5yAF93xcHtAc43bcl'

client = Client(api_key, api_secret, testnet=True)

default_symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'TRXUSDT', 
    'ADAUSDT', 'SHIBUSDT', 'LINKUSDT', 'DOTUSDT', 'BCHUSDT', 'LTCUSDT', 'MATICUSDT', 
    'UNIUSDT', 'XLMUSDT', 'AAVEUSDT', 'ARBUSDT', 'VETUSDT', 'ATOMUSDT'
]

def get_available_symbols():
    tickers = client.get_all_tickers()
    symbols = [ticker['symbol'] for ticker in tickers if ticker['symbol'].endswith('USDT')]
    return symbols
    
def calculate_price_change(df, num_coins):
    df['Price Change'] = df['Close'] - df['Open']
    df['Price Change (%)'] = (df['Price Change'] / df['Open'] * 100) / num_coins
    df['Color'] = df['Price Change'].apply(lambda x: 'green' if x >= 0 else 'red')
    return df

def calculate_direction_comparison(avg_changes_df, intervals):
    direction_comparison = {interval: {'Same Direction': 0, 'Opposite Direction': 0, 'Symbols': []} for interval in intervals}

    for interval in intervals:
        interval_data = avg_changes_df[avg_changes_df['Interval'] == interval]
        if len(interval_data) > 0:
            same_direction_count = (interval_data['Average Change (%)'] > 0).sum()
            opposite_direction_count = (interval_data['Average Change (%)'] <= 0).sum()
            
            direction_comparison[interval]['Same Direction'] = (same_direction_count / len(interval_data)) * 100
            direction_comparison[interval]['Opposite Direction'] = (opposite_direction_count / len(interval_data)) * 100

            # Determine whether to collect symbols from Same or Opposite Direction
            if direction_comparison[interval]['Opposite Direction'] < 50:
                # Collect symbols where Average Change (%) is positive
                symbols_to_collect = interval_data[interval_data['Average Change (%)'] <= 0]['Symbol'].tolist()
            else:
                # Collect symbols where Average Change (%) is negative
                symbols_to_collect = interval_data[interval_data['Average Change (%)'] >= 0]['Symbol'].tolist()
                
            direction_comparison[interval]['Symbols'] = symbols_to_collect

    return direction_comparison

def plot_comparison_chart(avg_changes, title):
    fig = go.Figure()

    for symbol in avg_changes['Symbol'].unique():
        symbol_data = avg_changes[avg_changes['Symbol'] == symbol]
        fig.add_trace(go.Bar(
            x=symbol_data['Interval'],
            y=symbol_data['Average Change (%)'],
            name=symbol
        ))

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
            l=100,
            r=100,
            t=100,
            b=100
        )
    )
    
    return fig

def plot_overall_average_chart(avg_changes, title):
    fig = go.Figure()

    avg_overall_change = avg_changes['Average Change (%)'].mean()
    
    fig.add_trace(go.Bar(
        x=['All Selected Coins'],
        y=[avg_overall_change],
        marker_color='purple',
        name='Overall Average Change (%)',
        text=[f'{avg_overall_change:.2f}%'],
        textposition='inside',
        textfont=dict(size=80)
    ))

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
            l=100,
            r=100,
            t=100,
            b=100
        )
    )
    
    return fig

def log_low_percentage_changes(direction_comparison, filename='logData.csv', threshold=50):
    # Open the CSV file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        # writer.writerow(['Interval', 'Direction', 'Percentage Change', 'Symbols'])
        
        for interval in direction_comparison.keys():
            for direction in ['Same Direction', 'Opposite Direction']:
                percentage_change = direction_comparison[interval][direction]
                if percentage_change < threshold:
                    symbols = direction_comparison[interval].get('Symbols', [])
                    # Write a row for each entry with low percentage change
                    writer.writerow([interval, direction, percentage_change, ', '.join(symbols)])

def plot_direction_comparison_chart(direction_comparison, title):
    fig = go.Figure()

    intervals = ['1m', '5m', '15m', '30m', '1h']
    interval_positions = {interval: i * 2 for i, interval in enumerate(intervals)}

    for interval in intervals:
        fig.add_trace(go.Bar(
            x=[f'{interval} - Same Direction'],
            y=[direction_comparison[interval]['Same Direction']],
            marker_color='green',
            name='Same Direction'
        ))

        fig.add_trace(go.Bar(
            x=[f'{interval} - Opposite Direction'],
            y=[-direction_comparison[interval]['Opposite Direction']],  # Negative for downward bar
            marker_color='red',
            name='Opposite Direction'
        ))

    shapes = []
    for position in interval_positions.values():
        shapes.append(dict(
            type='line',
            x0=position + 1.5,
            y0=-max(max([direction_comparison[interval]['Same Direction'] for interval in intervals]),
                    max([direction_comparison[interval]['Opposite Direction'] for interval in intervals])) - 10,
            x1=position + 1.5,
            y1=max(max([direction_comparison[interval]['Same Direction'] for interval in intervals]),
                    max([direction_comparison[interval]['Opposite Direction'] for interval in intervals])) + 10,
            line=dict(color='red', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Interval',
        yaxis_title='Percentage (%)',
        xaxis=dict(
            tickvals=[i + 1 for i in interval_positions.values()],
            ticktext=intervals,
            title_font=dict(size=18),  # Increase font size for x-axis title
            tickfont=dict(size=20),  # Increase font size for x-axis ticks
        ),
        yaxis=dict(
            title_font=dict(size=18),  # Increase font size for y-axis title
            tickfont=dict(size=20),  # Increase font size for y-axis ticks
        ),
        xaxis_rangeslider_visible=False,
        barmode='relative',
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,
            r=100,
            t=100,
            b=150  # Increased bottom margin to accommodate vertical text labels
        ),
        showlegend=False,
        shapes=shapes,
    )

    fig.update_traces(
        texttemplate='%{y:.2f}%',
        textposition='inside',
        textfont=dict(size=18)
    )

    fig.update_yaxes(
        range=[
            -max(max([direction_comparison[interval]['Same Direction'] for interval in intervals]),
                 max([direction_comparison[interval]['Opposite Direction'] for interval in intervals])) - 10,
            max(max([direction_comparison[interval]['Same Direction'] for interval in intervals]),
                max([direction_comparison[interval]['Opposite Direction'] for interval in intervals])) + 10
        ]
    )

    # Add text annotations for symbols with negative or positive changes
    annotations = []
    for i, interval in enumerate(intervals):
        if direction_comparison[interval]['Symbols']:
            if direction_comparison[interval]['Opposite Direction'] < 50:  # Taken from Opposite Direction
                y_position = -direction_comparison[interval]['Opposite Direction'] - 20*2  # Below the red bar
            else:  # Taken from Same Direction
                y_position = direction_comparison[interval]['Same Direction'] + 4*2  # Above the green bar

            symbols_text = '<br>'.join(direction_comparison[interval]['Symbols'])  # Join symbols with line breaks
            annotations.append(dict(
                x=i * 2 + 1,
                y=y_position,
                text=symbols_text,
                showarrow=False,
                font=dict(size=12, color='red' if direction_comparison[interval]['Opposite Direction'] < 50 else 'green'),
                align='center',
                xanchor='center'
            ))

    fig.update_layout(annotations=annotations)
    return fig

def plot_symbol_comparison_chart(symbol1, symbol2, intervals, title, smoothing_period=3):
    fig = go.Figure()

    interval_positions = {}
    current_position = 0

    percentage_changes = {symbol1: [], symbol2: []}

    for interval in intervals:
        for symbol in [symbol1, symbol2]:
            candles = client.get_klines(symbol=symbol, interval=interval)
            data = []
            for candle in candles:
                open_time = datetime.datetime.fromtimestamp(candle[0] / 1000)
                open_price = float(candle[1])
                close_price = float(candle[4])
                percentage_change = ((open_price - close_price) / open_price) * 100  # Adjusted formula
                data.append([open_time, open_price, close_price, percentage_change])
            
            df = pd.DataFrame(data, columns=['Time', 'Open', 'Close', 'Percentage Change'])
            df['Smoothed Change'] = df['Percentage Change'].rolling(window=smoothing_period).mean()  # Smoothing

            avg_change = df['Smoothed Change'].iloc[-1]  # Use the last smoothed value
            percentage_changes[symbol].append(avg_change)
        
        color1 = 'green' if symbol1 == 'BTCUSDT' else 'orange'
        color2 = 'yellow' if symbol2 == 'BCHUSDT' else 'orange'
        
        fig.add_trace(go.Bar(
            x=[f'{interval} - {symbol1}'],
            y=[percentage_changes[symbol1][-1]],
            name=f'{symbol1}',
            marker_color=color1,
            text=[f'{percentage_changes[symbol1][-1]:.4f}%'],
            textposition='inside',
            textfont=dict(size=40)  # Increased font size for percentage text
        ))

        fig.add_trace(go.Bar(
            x=[f'{interval} - {symbol2}'],
            y=[percentage_changes[symbol2][-1]],
            name=f'{symbol2}',
            marker_color=color2,
            text=[f'{percentage_changes[symbol2][-1]:.4f}%'],
            textposition='inside',
            textfont=dict(size=40)  # Increased font size for percentage text
        ))

        interval_positions[interval] = current_position
        current_position += 2

    all_changes = percentage_changes[symbol1] + percentage_changes[symbol2]
    y_max = max(all_changes, default=0) * 1.1
    y_min = min(all_changes, default=0) * 1.1
    y_min = min(y_min, 0)

    shapes = []
    for interval, position in interval_positions.items():
        shapes.append(dict(
            type='line',
            x0=position + 1.5,
            y0=y_min,
            x1=position + 1.5,
            y1=y_max - (y_max * 0.1),
            line=dict(color='red', width=2, dash='dash')
        ))

    # Add interval labels as annotations at the top of the graph
    annotations = []
    for interval, position in interval_positions.items():
        annotations.append(dict(
            x=position + 0.5,
            y=y_max,
            text=interval,
            showarrow=False,
            font=dict(size=20, color='white'),
            xanchor='center',
            yanchor='bottom',  # Anchor the text to the bottom of the annotation
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            showticklabels=False,  # Hide the default x-axis labels
            title_font=dict(size=18),  # Increase font size for x-axis title
            tickfont=dict(size=20),  # Increase font size for x-axis ticks
        ),
        yaxis_title='Percentage Change (Open - Close) (%)',  # Adjusted axis title
        barmode='group',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        autosize=True,
        height=800,
        width=1200,
        margin=go.layout.Margin(
            l=100,
            r=100,
            t=150,  # Increased top margin to accommodate the interval labels
            b=100
        ),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        yaxis=dict(
            range=[y_min, y_max],
            title_font=dict(size=18),  # Increase font size for x-axis title
            tickfont=dict(size=20),  # Increase font size for x-axis ticks
        )
    )

    fig.update_yaxes(autorange=True)

    return fig

def calculate_percentage_change(symbol, interval='1m'):
    # Retrieve the latest candles (price data) for the symbol
    candles = client.get_klines(symbol=symbol, interval=interval)

    percentage_changes = []

    # Calculate the percentage change for each interval (each candle)
    for candle in candles:
        open_price = float(candle[1])
        close_price = float(candle[4])
        percentage_change = ((close_price - open_price) / open_price) * 100
        percentage_changes.append(percentage_change)

    # Return only the most recent percentage change for the latest interval
    return percentage_changes[-1] if percentage_changes else 0

def plot_combined_percentage_chart(selected_symbols, title):
    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = []

    # Define the Asia/Jakarta time zone
    jakarta_tz = pytz.timezone('Asia/Jakarta')

    # Get current time in Asia/Jakarta time zone
    current_time = datetime.datetime.now(jakarta_tz)
    
    avg_percentage_change = 0
    if selected_symbols:
        avg_percentage_change = sum(calculate_percentage_change(symbol) for symbol in selected_symbols) / len(selected_symbols)

    # Append new data
    st.session_state.time_series_data.append({
        'Time': current_time,
        'Average Percentage Change': avg_percentage_change
    })

    # Keep only the latest 60 entries
    if len(st.session_state.time_series_data) > 60:
        st.session_state.time_series_data = st.session_state.time_series_data[-60:]

    df = pd.DataFrame(st.session_state.time_series_data)

    # Check if the 'Time' column is timezone-aware
    if df['Time'].dt.tz is None:
        # If not timezone-aware, localize to 'Asia/Jakarta'
        df['Time'] = df['Time'].dt.tz_localize(jakarta_tz)
    else:
        # If already timezone-aware, convert to 'Asia/Jakarta'
        df['Time'] = df['Time'].dt.tz_convert(jakarta_tz)

    fig = go.Figure()

    # Add trace for average percentage change
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average Percentage Change'],
        mode='lines+markers+text',
        name='Average Change',
        line=dict(color='blue'),
        text=[f"{pct:.2f}%" for pct in df['Average Percentage Change']],
        textposition='top center'
    ))

    # Add horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=df['Time'].min(),
        x1=df['Time'].max(),
        y0=0,
        y1=0,
        line=dict(color='red', width=2),
        xref='x',
        yref='y'
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time (Interval)',
        yaxis_title='Average Percentage Change (%)',
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,
            r=100,
            t=100,
            b=100
        ),
        font=dict(
            size=24  # Font size for the chart
        ),
        xaxis=dict(
            tickvals=df['Time'],
            ticktext=df['Time'].dt.strftime('%H:%M:%S'),
            title_font=dict(size=18),  # Font size for x-axis title
            tickfont=dict(size=20),  # Font size for x-axis ticks
        ),
        yaxis=dict(
            title_font=dict(size=18),  # Font size for y-axis title
            tickfont=dict(size=20),  # Font size for y-axis ticks
        ),
    )

    # Auto-adjust the y-axis range
    fig.update_yaxes(autorange=True)

    return fig

def plot_combined_percentage_chart_BAR(selected_symbols, title):
    if not selected_symbols:
        st.error("Please select at least one symbol.")
        return

    # Initialize time_series_data if it doesn't exist
    if 'time_series_data1' not in st.session_state:
        st.session_state.time_series_data1 = []

    intervals = ['1m', '5m', '15m', '30m', '1h']
    interval_labels = ['1 min', '5 min', '15 min', '30 min', '1 hour']
    avg_percentage_changes = []

    for interval in intervals:
        avg_percentage_change = 0
        if selected_symbols:
            avg_percentage_change = sum(calculate_percentage_change(symbol, interval) for symbol in selected_symbols) / len(selected_symbols)
        avg_percentage_changes.append(avg_percentage_change)
    # Define the Asia/Jakarta time zone
    jakarta_tz = pytz.timezone('Asia/Jakarta')

    # Get current time in Asia/Jakarta time zone
    current_time = datetime.datetime.now(jakarta_tz)

    # Store the time of the data collection for display purposes
    st.session_state.time_series_data1.append({
        'Time': current_time,
        'Average Percentage Change': avg_percentage_changes
    })

    # Prepare the data for plotting
    df = pd.DataFrame({
        'Interval': interval_labels,
        'Average Percentage Change': avg_percentage_changes
    })

    # Create a bar chart using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['Interval'],
        y=df['Average Percentage Change'],
        name='Average Change',
        marker_color='blue',
        text=[f"{pct:.4f}%" for pct in df['Average Percentage Change']],
        textposition='outside'
    ))

    # Add a horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=-0.5,  # Start just before the first bar
        x1=len(intervals) - 0.5,  # End just after the last bar
        y0=0,
        y1=0,
        line=dict(color='red', width=2),
        xref='x',
        yref='y'
    )

    # Update chart layout
    fig.update_layout(
        title=title,
        xaxis_title='Interval',
        yaxis_title='Average Percentage Change (%)',
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,
            r=100,
            t=100,
            b=100
        ),
        font=dict(
            size=24  # Font size for the chart
        ),
        xaxis=dict(
            title_font=dict(size=18),  # Font size for x-axis title
            tickfont=dict(size=20),  # Font size for x-axis ticks
        ),
        yaxis=dict(
            title_font=dict(size=18),  # Font size for y-axis title
            tickfont=dict(size=20),  # Font size for y-axis ticks
        ),
    )

    # Auto-adjust the y-axis range
    fig.update_yaxes(autorange=True)

    return fig

def calculate_percentage_change1(symbol, interval='5m'):
    candles = client.get_klines(symbol=symbol, interval=interval)
    total_percentage_change = 0
    total_data_points = 0

    for candle in candles:
        open_price = float(candle[1])
        close_price = float(candle[4])
        percentage_change = ((close_price - open_price) / open_price) * 100
        total_percentage_change += percentage_change
        total_data_points += 1

    return total_percentage_change / total_data_points if total_data_points > 0 else 0

def plot_combined_percentage_chart1(selected_symbols, title):
    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = []

    avg_percentage_change = 0
    if selected_symbols:
        avg_percentage_change = sum(calculate_percentage_change1(symbol) for symbol in selected_symbols) / len(selected_symbols)

    st.session_state.time_series_data.append({
        'Time': datetime.datetime.now(),
        'Average Percentage Change': avg_percentage_change
    })

    df = pd.DataFrame(st.session_state.time_series_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average Percentage Change'],
        mode='lines+markers+text',
        name='Average Change',
        line=dict(color='blue'),
        text=[f"{pct:.4f}%" for pct in df['Average Percentage Change']],
        textposition='top center'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time (Interval)',
        yaxis_title='Average Percentage Change (%)',
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,
            r=100,
            t=100,
            b=100
        ),
        font=dict(
            size=24  # General font size for the chart
        ),
        xaxis=dict(
            tickvals=df['Time'],
            ticktext=df['Time'].dt.strftime('%H:%M:%S'),
            title_font=dict(size=18),  # Increase font size for x-axis title
            tickfont=dict(size=20),  # Increase font size for x-axis ticks
        ),
        yaxis=dict(
            title_font=dict(size=18),  # Increase font size for y-axis title
            tickfont=dict(size=20),  # Increase font size for y-axis ticks
        ),
    )

    fig.update_yaxes(autorange=True)

    return fig

def calculate_percentage_change2(symbol, interval='15m'):
    candles = client.get_klines(symbol=symbol, interval=interval)
    total_percentage_change = 0
    total_data_points = 0

    for candle in candles:
        open_price = float(candle[1])
        close_price = float(candle[4])
        percentage_change = ((close_price - open_price) / open_price) * 100
        total_percentage_change += percentage_change
        total_data_points += 1

    return total_percentage_change / total_data_points if total_data_points > 0 else 0

def plot_combined_percentage_chart2(selected_symbols, title):
    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = []

    avg_percentage_change = 0
    if selected_symbols:
        avg_percentage_change = sum(calculate_percentage_change2(symbol) for symbol in selected_symbols) / len(selected_symbols)

    st.session_state.time_series_data.append({
        'Time': datetime.datetime.now(),
        'Average Percentage Change': avg_percentage_change
    })

    df = pd.DataFrame(st.session_state.time_series_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average Percentage Change'],
        mode='lines+markers+text',
        name='Average Change',
        line=dict(color='blue'),
        text=[f"{pct:.4f}%" for pct in df['Average Percentage Change']],
        textposition='top center'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time (Interval)',
        yaxis_title='Average Percentage Change (%)',
        template='plotly_dark',
        autosize=True,
        height=600,
        width=1200,
        margin=go.layout.Margin(
            l=100,
            r=100,
            t=100,
            b=100
        ),
        font=dict(
            size=24  # General font size for the chart
        ),
        xaxis=dict(
            tickvals=df['Time'],
            ticktext=df['Time'].dt.strftime('%H:%M:%S'),
            title_font=dict(size=18),  # Increase font size for x-axis title
            tickfont=dict(size=20),  # Increase font size for x-axis ticks
        ),
        yaxis=dict(
            title_font=dict(size=18),  # Increase font size for y-axis title
            tickfont=dict(size=20),  # Increase font size for y-axis ticks
        ),
    )

    fig.update_yaxes(autorange=True)

    return fig

def main():
    # load_logs()
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Compare 20 Coins", "Compare BTCUSDT and BCHUSDT"], captions = ["[Log 5 minutes](https://cryptoanalyzelog5.streamlit.app/) | [Log 15 minutes](https://cryptoanalyzelog15.streamlit.app/)",""])

    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = []

    elif selection == "log 15 menit":
        st.title("Cryptocurrency Price Analysis")

        symbols = get_available_symbols()

        # selected_symbols = st.multiselect(
        #     "Select up to 20 symbols", 
        #     options=symbols,
        #     max_selections=20,
        #     default=st.session_state.selected_symbols
        # )

        selected_symbols = st.multiselect(
            'Select coins to compare:', 
            options=get_available_symbols(), 
            max_selections=20,
            default=default_symbols
        )

        if selected_symbols != st.session_state.selected_symbols:
            st.session_state.selected_symbols = selected_symbols

        if len(selected_symbols) < 2:
            st.error("Please select at least 2 symbols to compare.")
            return

        intervals = ['1m', '5m', '15m', '30m', '1h']
        num_coins = len(selected_symbols)

        while True:
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
                    df = calculate_price_change(df, num_coins)
                    
                    avg_change = df['Price Change (%)'].mean()
                    avg_changes.append({'Symbol': symbol, 'Interval': interval, 'Average Change (%)': avg_change})

            avg_changes_df = pd.DataFrame(avg_changes)

            if len(selected_symbols) > 1:
                
                direction_comparison = calculate_direction_comparison(avg_changes_df, intervals)
                fig_direction_comparison = plot_direction_comparison_chart(direction_comparison, "Direction Comparison (%) of Selected Coins")
                # st.plotly_chart(fig_direction_comparison, use_container_width=True)
                
                avg_changes_df = pd.DataFrame(avg_changes)

                fig_combined2 = plot_combined_percentage_chart2(selected_symbols, "Combined Average Percentage Change for Selected Coins")
                st.plotly_chart(fig_combined2, use_container_width=True)

                # fig_combined1 = plot_combined_percentage_chart1(selected_symbols, "Combined Average Percentage Change for Selected Coins 5m")
                # st.plotly_chart(fig_combined1, use_container_width=True)

                # fig_comparison = plot_comparison_chart(avg_changes_df, "Average Price Change (%) by Interval and Symbol")
                # st.plotly_chart(fig_comparison, use_container_width=True)

            time.sleep(900)  # Wait for 30 seconds before updating
            st.rerun()  # Rerun the script to update data

    elif selection == "log 5 menit":
        url = "https://cryptoanalyzelog5.streamlit.app/"
        st.markdown(f'<a href="{url}" target="_blank">Click here to visit Example.com</a>', unsafe_allow_html=True)

    if selection == "Compare 20 Coins":
        st.title("Cryptocurrency Price Analysis")

        symbols = get_available_symbols()

        # selected_symbols = st.multiselect(
        #     "Select up to 20 symbols", 
        #     options=symbols,
        #     max_selections=20,
        #     default=st.session_state.selected_symbols
        # )

        selected_symbols = st.multiselect(
            'Select coins to compare:', 
            options=get_available_symbols(), 
            max_selections=20,
            default=default_symbols
        )

        if selected_symbols != st.session_state.selected_symbols:
            st.session_state.selected_symbols = selected_symbols

        if len(selected_symbols) < 2:
            st.error("Please select at least 2 symbols to compare.")
            return

        intervals = ['1m', '5m', '15m', '30m', '1h']
        num_coins = len(selected_symbols)

        while True:
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
                    df = calculate_price_change(df, num_coins)
                    
                    avg_change = df['Price Change (%)'].mean()
                    avg_changes.append({'Symbol': symbol, 'Interval': interval, 'Average Change (%)': avg_change})

            avg_changes_df = pd.DataFrame(avg_changes)

            if len(selected_symbols) > 1:
                
                # direction_comparison = calculate_direction_comparison(avg_changes_df, intervals)
                # fig_direction_comparison = plot_direction_comparison_chart(direction_comparison, "Direction Comparison (%) of Selected Coins")
                # # st.plotly_chart(fig_direction_comparison, use_container_width=True)
                
                avg_changes_df = pd.DataFrame(avg_changes)

                fig_combined = plot_combined_percentage_chart(selected_symbols, "Line Combined Average Percentage Change for Selected Coins")
                st.plotly_chart(fig_combined, use_container_width=True)

                fig_combined1 = plot_combined_percentage_chart_BAR(selected_symbols, "Bar Combined Average Percentage Change for Selected Coins")
                st.plotly_chart(fig_combined1, use_container_width=True)

                fig_comparison = plot_comparison_chart(avg_changes_df, "Average Price Change (%) by Interval and Symbol")
                st.plotly_chart(fig_comparison, use_container_width=True)

            time.sleep(30)  # Wait for 30 seconds before updating
            st.rerun()  # Rerun the script to update data

    elif selection == "Compare BTCUSDT and BCHUSDT":
        st.title("Compare BTCUSDT and BCHUSDT")

        intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '8h', '1d']

        log_entries = []  # Initialize log entries list

        while True:
            fig_comparison = plot_symbol_comparison_chart('BTCUSDT', 'BCHUSDT', intervals, "BTCUSDT vs BCHUSDT Price Change (%)")
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Display logs in a table
            # if datetime.datetime.now() - st.session_state.last_log_time >= datetime.timedelta(minutes=15):
            #     if os.path.exists('logs.csv'):
            #         logs_df = pd.read_csv('logs.csv', names=['Log'])
            #         st.write("### Logs")
            #         st.write(logs_df)

            #         # Provide download link for the CSV file
            #         with open('logs.csv', 'r') as file:
            #             st.download_button(
            #                 label="Download Logs",
            #                 data=file,
            #                 file_name='logs.csv',
            #                 mime='text/csv'
            #             )

            time.sleep(30)  # Wait for 30 seconds before updating
            st.rerun()  # Rerun the script to update data


if __name__ == "__main__":
    main()