from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'outputs' / 'restaurant_predictions.csv'
PROFILE_PATH = ROOT / 'outputs' / 'cluster_profiles.csv'

st.set_page_config(page_title='Restaurant Growth Potential', page_icon='🍽️', layout='wide')

@st.cache_data
def load_results():
    df = pd.read_csv(DATA_PATH)
    profiles = pd.read_csv(PROFILE_PATH) if PROFILE_PATH.exists() else pd.DataFrame()
    return df, profiles

if not DATA_PATH.exists():
    st.error('Model output not found. Run: python train_model.py')
    st.stop()

df, profiles = load_results()

st.title('Restaurant Growth Potential Modeling & Strategic Classification')
st.caption('SkyCity Auckland Restaurants & Bars | Unsupervised clustering + transparent GPI scoring')

# Sidebar filters
st.sidebar.header('Filters')
cuisine = st.sidebar.multiselect('Cuisine', sorted(df['CuisineType'].dropna().unique()), default=[])
segment = st.sidebar.multiselect('Segment', sorted(df['Segment'].dropna().unique()), default=[])
subregion = st.sidebar.multiselect('Subregion', sorted(df['Subregion'].dropna().unique()), default=[])
cluster_names = st.sidebar.multiselect('Cluster', sorted(df['ClusterName'].dropna().unique()), default=[])

filtered = df.copy()
if cuisine: filtered = filtered[filtered['CuisineType'].isin(cuisine)]
if segment: filtered = filtered[filtered['Segment'].isin(segment)]
if subregion: filtered = filtered[filtered['Subregion'].isin(subregion)]
if cluster_names: filtered = filtered[filtered['ClusterName'].isin(cluster_names)]

# Navigation
page = st.sidebar.radio('Section', [
    'Executive Dashboard','Restaurant Explorer','Cluster Analysis',
    'Growth Potential','Strategic Recommendations'
])

if page == 'Executive Dashboard':
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric('Restaurants', f'{len(filtered):,}')
    c2.metric('Average GPI', f'{filtered.GPI.mean():.1f}')
    c3.metric('Very High / High', f'{filtered.GPI_Category.isin(["Very High Potential","High Potential"]).sum():,}')
    c4.metric('Low Potential', f'{(filtered.GPI_Category == "Low Potential").sum():,}')
    c5.metric('Avg Growth', f'{filtered.GrowthMomentum.mean()*100:.2f}%')
    c6.metric('Avg Profit Margin', f'{filtered.ProfitMargin.mean()*100:.2f}%')

    left,right = st.columns(2)
    with left:
        fig = px.bar(filtered['ClusterName'].value_counts().reset_index(), x='ClusterName', y='count', title='Cluster Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(filtered, x='GPI', nbins=20, title='Growth Potential Index Distribution')
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(filtered, x='TotalRevenue', y='TotalNetProfit', size='MonthlyOrders', color='GPI_Category', hover_name='RestaurantName', title='Revenue vs Net Profit')
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Restaurant Explorer':
    names = sorted(filtered['RestaurantName'].unique())
    if not names:
        st.warning('No restaurants match the selected filters.')
        st.stop()
    selected = st.selectbox('Select restaurant', names)
    r = filtered[filtered['RestaurantName'] == selected].iloc[0]
    st.subheader(r['RestaurantName'])
    a,b,c,d = st.columns(4)
    a.metric('GPI', f"{r['GPI']:.1f}")
    b.metric('Category', r['GPI_Category'])
    c.metric('Cluster', r['ClusterName'])
    d.metric('Recommendation', r['Recommendation'])

    st.write(f"**Cuisine:** {r['CuisineType']}  |  **Segment:** {r['Segment']}  |  **Subregion:** {r['Subregion']}")
    metrics = pd.DataFrame({
        'Metric':['Monthly Orders','AOV','Total Revenue','Net Profit','Profit Margin','Growth','Aggregator Dependence','Delivery Radius'],
        'Value':[f"{r.MonthlyOrders:,.0f}",f"${r.AOV:,.2f}",f"${r.TotalRevenue:,.2f}",f"${r.TotalNetProfit:,.2f}",f"{r.ProfitMargin*100:.2f}%",f"{r.GrowthMomentum*100:.2f}%",f"{r.AggregatorDependence*100:.1f}%",f"{r.DeliveryRadiusKM:.1f} km"]
    })
    st.table(metrics)

    radar_categories = ['Growth','Profitability','Cost Resilience','Channel Balance','Scalability']
    radar_values = [r.GPI_Growth,r.GPI_Profitability,r.GPI_CostResilience,r.GPI_ChannelBalance,r.GPI_Scalability]
    fig = go.Figure(go.Scatterpolar(r=radar_values + [radar_values[0]], theta=radar_categories + [radar_categories[0]], fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False, title='GPI Component Scores')
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Cluster Analysis':
    st.subheader('Restaurant Cluster Analysis')
    fig = px.scatter(filtered, x='PCA1', y='PCA2', color='ClusterName', hover_name='RestaurantName', hover_data=['CuisineType','Segment','Subregion','GPI','Recommendation'], title='PCA Cluster Map')
    st.plotly_chart(fig, use_container_width=True)
    if not profiles.empty:
        st.dataframe(profiles, use_container_width=True)

elif page == 'Growth Potential':
    st.subheader('Growth Potential Analysis')
    top = filtered.nlargest(10, 'GPI')[['RestaurantName','CuisineType','GPI','GPI_Category','ClusterName','Recommendation']]
    bottom = filtered.nsmallest(10, 'GPI')[['RestaurantName','CuisineType','GPI','GPI_Category','ClusterName','Recommendation']]
    l,r = st.columns(2)
    with l:
        st.write('Top 10 Restaurants')
        st.dataframe(top, use_container_width=True, hide_index=True)
    with r:
        st.write('Bottom 10 Restaurants')
        st.dataframe(bottom, use_container_width=True, hide_index=True)
    fig = px.bar(filtered['GPI_Category'].value_counts().reset_index(), x='GPI_Category', y='count', title='GPI Categories')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader('Strategic Recommendations')
    counts = filtered['Recommendation'].value_counts().reset_index()
    fig = px.pie(counts, names='Recommendation', values='count', title='Recommendation Mix')
    st.plotly_chart(fig, use_container_width=True)
    selected = st.selectbox('Select restaurant for recommendation', sorted(filtered['RestaurantName'].unique()))
    r = filtered[filtered['RestaurantName'] == selected].iloc[0]
    st.markdown(f'### {r.RestaurantName}')
    st.info(f"**Recommended strategy: {r.Recommendation}**")
    st.write(f"**GPI:** {r.GPI:.1f}/100  |  **Cluster:** {r.ClusterName}")
    drivers = []
    risks = []
    if r.GrowthMomentum > 0.03: drivers.append('Strong positive growth momentum')
    if r.ProfitMargin > 0.15: drivers.append('Healthy profit margin')
    if r.Scale >= filtered.Scale.median(): drivers.append('Above-median business scale')
    if r.DeliveryRadiusKM >= filtered.DeliveryRadiusKM.median(): drivers.append('Good delivery reach')
    if r.AggregatorDependence >= 0.70: risks.append('High third-party aggregator dependence')
    if r.CostPressure >= filtered.CostPressure.median(): risks.append('Above-median cost pressure')
    if r.TotalNetProfit < 0: risks.append('Negative total net profit')
    st.write('**Main growth drivers**')
    for x in drivers or ['No dominant positive driver identified; maintain close monitoring.']:
        st.write('• ' + x)
    st.write('**Main risks**')
    for x in risks or ['No major risk signal identified from the selected indicators.']:
        st.write('• ' + x)
