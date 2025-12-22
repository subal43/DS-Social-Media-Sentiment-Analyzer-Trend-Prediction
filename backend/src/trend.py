import streamlit as st
import plotly.express as px


def sentiment_trend_analysis(data):
    """
    Displays interactive sentiment trend analysis with:
    - Daily / Weekly / Monthly aggregation
    - Separate charts per sentiment
    - Optional moving average smoothing
    """

    st.subheader("Sentiment Trend Analysis")

    time_freq = st.selectbox(
        "Select Time Aggregation",
        ["Daily", "Weekly", "Monthly"]
    )
    use_moving_avg = st.checkbox("Show Moving Average")

    if time_freq == "Weekly":
        data['Period'] = data['Date'].dt.to_period("W").dt.start_time
    elif time_freq == "Monthly":
        data['Period'] = data['Date'].dt.to_period("M").dt.start_time
    else:
        data['Period'] = data['Date']

    trend_data = (
        data.groupby(['Period', 'Sentiment']).size().reset_index(name='Count')
    )

    for sentiment in trend_data['Sentiment'].unique():
        sentiment_df = trend_data[trend_data['Sentiment'] == sentiment].copy()

        fig = px.line(
            sentiment_df,
            x='Period',
            y='Count',
            title=f"{sentiment} Sentiment Trend",
            markers=True
        )
        if use_moving_avg:
            sentiment_df['Moving_Avg'] = sentiment_df['Count'].rolling(window=3).mean()

            fig.add_scatter(
                x=sentiment_df['Period'],
                y=sentiment_df['Moving_Avg'],
                mode='lines',
                name='Moving Average'
            )

        st.plotly_chart(fig, use_container_width=True)

    
    