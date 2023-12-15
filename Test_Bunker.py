import streamlit as st
st.set_page_config(layout="wide")
st.write('project updated on 20231214')



import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from dateutil import relativedelta
from pandas.tseries.offsets import BDay
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Getting Bunker Data
st.text('----Getting Bunker Data...')

@st.cache_data(ttl=43200)
def load_bunker_data():
    bunker=pd.read_excel('Historical_data_-_Bunker_Prices.xlsx')

    bunker_s=bunker[bunker['Period']=='SPOT']
    bunker_f=bunker[bunker['Period']!='SPOT']

    bunker_f[['Month','Year']]=bunker_f['Period'].str.split('-',expand=True)
    bunker_f['Month'].replace({'JAN':'1','FEB':'2','MAR':'3','APR':'4','MAY':'5','JUN':'6',
                         'JUL':'7','AUG':'8','SEP':'9','OCT':'10','NOV':'11','DEC':'12'},inplace=True)

    bunker_f['Fixed Contract']=bunker_f['Year']+'_M'+bunker_f['Month']
    bunker_f['Month']=pd.to_numeric(bunker_f['Month'])
    bunker_f['Year']=pd.to_numeric(bunker_f['Year'])

    bunker_f['Archive Month']=bunker_f['Date'].dt.month
    bunker_f['Archive Year']=bunker_f['Date'].dt.year
    bunker_f['Rolling Month Gap']=(bunker_f['Year']-bunker_f['Archive Year'])*12+(bunker_f['Month']-bunker_f['Archive Month'])

    return bunker_s, bunker_f

bunker_s=load_bunker_data()[0]
bunker_f=load_bunker_data()[1]

if 'bunker_s' not in st.session_state:
    st.session_state['bunker_s']=bunker_s

if 'bunker_f' not in st.session_state:
    st.session_state['bunker_f']=bunker_f


st.text('Bunker Data Retrieved!...')

cutoff = pd.to_datetime('today')
curryear=cutoff.year

plot_ticks='inside'
plot_tickwidth=2
plot_ticklen=10
plot_title_font_color='dodgerblue'
plot_title_font_size=25
plot_legend_font_size=15
plot_axis=dict(tickfont = dict(size=15))

st.title('Bunker')
st.text('Dry Bulk Freight (Bunker) Interactive Dashboard')
st.write('Made by Huanyu')

st.markdown('## **Candle Chart**')

bunker_s=st.session_state['bunker_s']
bunker_f=st.session_state['bunker_f']
wti=pd.read_csv('WTI原油期货历史数据.csv')
brent=pd.read_csv('伦敦布伦特原油期货历史数据.csv')
wti.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','高':'High','低':'Low','交易量':'Volume','涨跌幅':'DoD'},inplace=True)
brent.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','高':'High','低':'Low','交易量':'Volume','涨跌幅':'DoD'},inplace=True)
wti_pt=wti[['Date','Close']]
wti_pt.set_index('Date',inplace=True)
wti_pt.rename(columns={'Close':'WTI'},inplace=True)
wti_pt.index=pd.to_datetime(wti_pt.index)
brent_pt=brent[['Date','Close']]
brent_pt.set_index('Date',inplace=True)
brent_pt.rename(columns={'Close':'Brent'},inplace=True)
brent_pt.index=pd.to_datetime(brent_pt.index)

brent['Date']=pd.to_datetime(brent['Date'])
brent_w=brent.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
brent_m=brent.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()

wti['Date']=pd.to_datetime(wti['Date'])
wti_w=wti.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
wti_m=wti.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()

cdtype=st.selectbox('Select Oil Type',options=['Brent','WTI'],key='cdtype')
yr1=st.number_input('Input Start Year',min_value=2015,max_value=curryear,value=curryear-1,step=1,key='yr1')
cdfreq=st.selectbox('Select Frequency',options=['Weekly','Daily','Monthly'],key='cdfreq')


if cdtype=='Brent':
    if cdfreq=='Daily':
        brent_d=brent[brent['Date'].dt.year>=yr1]
        cddata=brent_d
    elif cdfreq=='Weekly':
        brent_w=brent_w[brent_w['Date'].dt.year>=yr1]
        cddata=brent_w
    elif cdfreq=='Monthly':
        brent_m=brent_m[brent_m['Date'].dt.year>=yr1]
        cddata=brent_m

if cdtype=='WTI':
    if cdfreq=='Daily':
        wti_d=wti[wti['Date'].dt.year>=yr1]
        cddata=wti_d
    elif cdfreq=='Weekly':
        wti_w=wti_w[wti_w['Date'].dt.year>=yr1]
        cddata=wti_w
    elif cdfreq=='Monthly':
        wti_m=wti_m[wti_m['Date'].dt.year>=yr1]
        cddata=wti_m


candle=go.Figure(data=[go.Candlestick(x=cddata['Date'],open=cddata['Open'],high=cddata['High'],low=cddata['Low'],close=cddata['Close'])])
candle.update_layout(title=cdtype+' '+cdfreq+' Candle Chart',width=1000,height=500)
candle.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
candle.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
candle.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(candle)



st.markdown('## **Major Rolling Contracts**')
rolling_gap=st.selectbox('Select Rolling Month Gap (+n months)',options=[1,2],key='999')
tonsl=st.multiselect('Select Product in Tonnes',options=['RDM35FO','RDM_0.5','SING_0.5','SIN_380','WTI in Tonnes','Brent in Tonnes'],default=['RDM35FO','RDM_0.5','SING_0.5','SIN_380'],key='sl111')
barrelsl=st.multiselect('Select Product in Barrels',options=['WTI','Brent'],default=['Brent'],key='sl222')
rangelist=st.selectbox('Select Range',options=['Year to Date','Last Year to Date','Month to Date','All'])

today = pd.to_datetime('today')
if rangelist=='Last Year to Date':
    rangestart=date(today.year-1,1,1)
elif rangelist=='Month to Date':
    rangestart=date(today.year,today.month,1)
elif rangelist=='Year to Date':
    rangestart=date(today.year,1,1)
else:
    rangestart=date(2015,1,1)


bunker_major=bunker_f[bunker_f['Rolling Month Gap']==rolling_gap]
bunker_major_pt=bunker_major.pivot_table(index='Date',columns='Route',values='Amount',aggfunc='mean')
bunker_major_pt=pd.merge(bunker_major_pt,wti_pt,left_index=True,right_index=True,how='left')
bunker_major_pt=pd.merge(bunker_major_pt,brent_pt,left_index=True,right_index=True,how='left')
bunker_major_pt.sort_index(ascending=False,inplace=True)
bunker_major_pt['WTI in Tonnes']=bunker_major_pt['WTI']/0.136
bunker_major_pt['Brent in Tonnes']=bunker_major_pt['Brent']/0.136
bunker_major_pt=bunker_major_pt[bunker_major_pt.index>=pd.to_datetime(rangestart)]

subplot_fig = make_subplots(specs=[[{"secondary_y": True}]])
fig1=px.line(bunker_major_pt[tonsl])
fig2=px.line(bunker_major_pt[barrelsl])
fig2.update_traces(yaxis='y2')
subplot_fig.add_traces(fig1.data+fig2.data)
subplot_fig.update_layout(title='+'+str(rolling_gap)+'M Fuel Oil Rolling Contract and Continuous Crude Oil Contract Overview',width=1000,height=500)
subplot_fig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
subplot_fig.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
subplot_fig.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(subplot_fig)

st.markdown('#### **----Contract Spread**')
ct1=st.selectbox('Select Contract 1',options=['SING_0.5','RDM_0.5','SIN_380','RDM35FO','WTI in Tonnes','Brent in Tonnes','WTI','Brent'],key='9991')
ct2=st.selectbox('Select Contract 2',options=['Brent in Tonnes','WTI in Tonnes','Brent','WTI','RDM35FO','RDM_0.5','SING_0.5','SIN_380'],key='9992')

bunker_major_pt['Spread']=bunker_major_pt[ct1]-bunker_major_pt[ct2]
mspplot=px.line(bunker_major_pt['Spread'],width=1000,height=500,title=ct1+' Minus '+ct2 +' Spread (+'+str(rolling_gap)+'M Rolling Contract for Fuel Oil)')
mspplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
mspplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(mspplot)

st.markdown('## **Fuel Oil Price**')

type=st.selectbox('Select Type',options=['SING_0.5','SIN_380','RDM_0.5','RDM35FO'],key='000')

sing5_f=bunker_f[bunker_f['Route']==type]
sing5_s=bunker_s[bunker_s['Route']==type]
sing5_f.sort_values(by='Date',ascending=True,inplace=True)

sing5_pt1=sing5_f.pivot_table(index='Date',columns='Fixed Contract',values='Amount',aggfunc='mean')
sing5_pt1.index=pd.to_datetime(sing5_pt1.index,dayfirst=True)
sing5_pt1.sort_index(inplace=True)

sing5_pt2=sing5_f.pivot_table(index='Date',columns='Rolling Month Gap',values='Amount',aggfunc='mean')
sing5_pt2.sort_index(inplace=True)
sing5_pt2.columns=sing5_pt2.columns.astype('int64')


sing5_s.set_index('Date',inplace=True)
sing5_s.sort_index(ascending=True,inplace=True)
sing5_s=sing5_s[['Amount']]

sing5_s.rename(columns={'Amount':'Spot'},inplace=True)
sing5_s.drop_duplicates(inplace=True)


sing5_pt1=pd.merge(sing5_s,sing5_pt1,left_index=True,right_index=True,how='outer')
sing5_pt2=pd.merge(sing5_s,sing5_pt2,left_index=True,right_index=True,how='outer')
sing5_pt2.rename(columns={'Spot':0},inplace=True)


tday=sing5_pt1.index.max()
lday=tday-BDay(1)
l2day=tday-BDay(2)
l3day=tday-BDay(3)
l4day=tday-BDay(4)
lweek=tday-BDay(5)
l2week=tday-BDay(10)
l3week=tday-BDay(15)
lmonth=tday-BDay(20)
l2month=tday-BDay(45)

s0='Spot'

for k in range(30):
    exec(f'm{k}=str((tday + relativedelta.relativedelta(months=k)).year)+\'_M\'+str((tday + relativedelta.relativedelta(months=k)).month)')



st.header(type+' Summary')
sing5_df=sing5_pt1[['Spot',m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m15,m18,m21,m24]]

sing5_df.index=sing5_df.index.date

sing5_=pd.concat([sing5_df.loc[[tday.date()]],sing5_df.loc[[lday.date()]],sing5_df.loc[[lweek.date()]],sing5_df.loc[[lmonth.date()]]])
st.write(sing5_.style.format('{:,.0f}'))



st.markdown('#### **Change**')
sing5_.loc['DoD Chg']=sing5_df.loc[tday.date()]-sing5_df.loc[lday.date()]
sing5_.loc['WoW Chg']=sing5_df.loc[tday.date()]-sing5_df.loc[lweek.date()]
sing5_.loc['MoM Chg']=sing5_df.loc[tday.date()]-sing5_df.loc[lmonth.date()]
sing5_chg=pd.concat([sing5_.loc[['DoD Chg']],sing5_.loc[['WoW Chg']],sing5_.loc[['MoM Chg']]])
st.write(sing5_chg.style.format('{:,.0f}'))

st.markdown('#### **Change in Percentage**')
sing5_.loc['DoD Chg %']=sing5_.loc['DoD Chg']/sing5_df.loc[lday.date()]
sing5_.loc['WoW Chg %']=sing5_.loc['WoW Chg']/sing5_df.loc[lweek.date()]
sing5_.loc['MoM Chg %']=sing5_.loc['MoM Chg']/sing5_df.loc[lmonth.date()]
p4tc_chgpct=pd.concat([sing5_.loc[['DoD Chg %']],sing5_.loc[['WoW Chg %']],sing5_.loc[['MoM Chg %']]])
st.write(p4tc_chgpct.style.format('{:,.2%}'))


st.header(type+' Forward Contracts Line Chart')
st.markdown('#### **----Fixed Contracts**')
rangelist1=st.selectbox('Select Range',options=['Year to Date','Month to Date','Last Week to Date','All'],key='rg1')
sllist1=st.multiselect('Select Contracts',options=sing5_pt1.columns,default=['Spot',m1,m2,m3,m4,m5,m6,m9,m12,m24],key='sl1')
sing5_sl=sing5_pt1[sllist1]

today = pd.to_datetime('today')
if rangelist1=='Last Week to Date':
    rangestart1=today - timedelta(days=today.weekday()) + timedelta(days=6, weeks=-2)
elif rangelist1=='Month to Date':
    rangestart1=date(today.year,today.month,1)
elif rangelist1=='Year to Date':
    rangestart1=date(today.year,1,1)
else:
    rangestart1=date(2015,1,1)

sing5_sl=sing5_sl[pd.to_datetime(sing5_sl.index)>=pd.to_datetime(rangestart1)]
lplot=px.line(sing5_sl,width=1000,height=500,title=type+' Spot and Fixed Forward Contract Historical Price')
lplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
lplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(lplot)

st.markdown('#### **----Rolling Contracts**')

rangelist_r=st.selectbox('Select Range',options=['Year to Date','Month to Date','Last Week to Date','All'],key='101')
sllist_r=st.multiselect('Select Contracts (+n Months)',options=sing5_pt2.columns,default=[0,1,2,3,4,5,6,9,12,24],key='102')
sing5_sl=sing5_pt2[sllist_r]

today = pd.to_datetime('today')
if rangelist_r=='Last Week to Date':
    rangestart_r=today - timedelta(days=today.weekday()) + timedelta(days=6, weeks=-2)
elif rangelist_r=='Month to Date':
    rangestart_r=date(today.year,today.month,1)
elif rangelist_r=='Year to Date':
    rangestart_r=date(today.year,1,1)
else:
    rangestart_r=date(2015,1,1)

sing5_sl=sing5_sl[pd.to_datetime(sing5_sl.index)>=pd.to_datetime(rangestart_r)]
lplot=px.line(sing5_sl,width=1000,height=500,title=type+' Spot and Rolling Forward Contract Historical Price')
lplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
lplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(lplot)


st.header(type+' Technical Analysis')
st.markdown('#### **----Fixed Contracts**')

contractlist=st.selectbox('Select Spot or Forward Contract',options=list(sing5_pt1.columns))
bb=st.number_input('Bollinger Bands Window',value=20)
ma1=st.number_input('Short Term Moving Average Window',value=20)
ma2=st.number_input('Long Term Moving Average Window',value=50)


sing5_contract=sing5_pt1[[contractlist]]
sing5_contract.dropna(inplace=True)

sing5_contract.sort_index(inplace=True)
indicator_mast = SMAIndicator(close=sing5_contract[contractlist], window=ma1)
indicator_malt = SMAIndicator(close=sing5_contract[contractlist], window=ma2)
indicator_bb = BollingerBands(close=sing5_contract[contractlist], window=bb, window_dev=2)
sing5_contract['ma_st'] = indicator_mast.sma_indicator()
sing5_contract['ma_lt'] = indicator_malt.sma_indicator()
sing5_contract['bb_m'] = indicator_bb.bollinger_mavg()
sing5_contract['bb_h'] = indicator_bb.bollinger_hband()
sing5_contract['bb_l'] = indicator_bb.bollinger_lband()


contractplot=px.line(sing5_contract,width=1000,height=500,title=type+' '+contractlist+' Fixed Contract Bollinger Bands and Moving Average')
contractplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
contractplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(contractplot)


st.markdown('#### **----Rolling Contracts**')

rangelist_r=st.selectbox('Select Range',options=['Year to Date','Month to Date','Last Week to Date','All'],key='205')
contractlist_r=st.selectbox('Select Contracts (+n Months)',options=list(sing5_pt2.columns),key='201')
bb_r=st.number_input('Bollinger Bands Window',value=20,key='202')
ma1_r=st.number_input('Short Term Moving Average Window',value=20,key='203')
ma2_r=st.number_input('Long Term Moving Average Window',value=50,key='204')

if rangelist_r=='Last Week to Date':
    rangestart_r=today - timedelta(days=today.weekday()) + timedelta(days=6, weeks=-2)
elif rangelist_r=='Month to Date':
    rangestart_r=date(today.year,today.month,1)
elif rangelist_r=='Year to Date':
    rangestart_r=date(today.year,1,1)
else:
    rangestart_r=date(2015,1,1)


sing5_contract=sing5_pt2[[contractlist_r]]
sing5_contract.dropna(inplace=True)
sing5_contract=sing5_contract[pd.to_datetime(sing5_contract.index)>=pd.to_datetime(rangestart_r)]

sing5_contract.sort_index(inplace=True)
indicator_mast = SMAIndicator(close=sing5_contract[contractlist_r], window=ma1_r)
indicator_malt = SMAIndicator(close=sing5_contract[contractlist_r], window=ma2_r)
indicator_bb = BollingerBands(close=sing5_contract[contractlist_r], window=bb_r, window_dev=2)
sing5_contract['ma_st'] = indicator_mast.sma_indicator()
sing5_contract['ma_lt'] = indicator_malt.sma_indicator()
sing5_contract['bb_m'] = indicator_bb.bollinger_mavg()
sing5_contract['bb_h'] = indicator_bb.bollinger_hband()
sing5_contract['bb_l'] = indicator_bb.bollinger_lband()

contractplot=px.line(sing5_contract,width=1000,height=500,title=type+' +'+str(contractlist_r)+'M Rolling Contract Bollinger Bands and Moving Average')
contractplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
contractplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(contractplot)

st.header(type+' Spot and Rolling FFA Contracts Seasonality')
contractlist_r=st.selectbox('Select Contracts (+n Months)',options=list(sing5_pt2.columns),key='211')
freq=st.radio('Select Frequency',options=['Weekly','Monthly','Quarterly'],key='spotfreq')
sing5_sp=sing5_pt2[[contractlist_r]]
sing5_sp.index=pd.to_datetime(sing5_sp.index)

if freq=='Weekly':
    sing5_sp['Year']=sing5_sp.index.year
    sing5_sp['Week']=sing5_sp.index.isocalendar().week
    sing5_sp.loc[sing5_sp[sing5_sp.index.date==date(2016,1,2)].index,'Week']=0
    sing5_sp.loc[sing5_sp[sing5_sp.index.date==date(2021,1,2)].index,'Week']=0
    sing5_sp.loc[sing5_sp[sing5_sp.index.date==date(2022,1,1)].index,'Week']=0
    yrsl=st.multiselect('Select Years',options=sing5_sp['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear1')
    sing5_sp=sing5_sp[sing5_sp['Year'].isin(yrsl)]
    sing5_sppt=sing5_sp.pivot_table(index='Week',columns='Year',values=contractlist_r,aggfunc='mean')

    spotplot=px.line(sing5_sppt,width=1000,height=500,title=type+' +'+str(contractlist_r)+'M Rolling Contract Weekly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

elif freq=='Monthly':
    sing5_sp['Year']=sing5_sp.index.year
    sing5_sp['Month']=sing5_sp.index.month
    yrsl=st.multiselect('Select Years',options=sing5_sp['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear2')
    sing5_sp=sing5_sp[sing5_sp['Year'].isin(yrsl)]
    sing5_sppt=sing5_sp.pivot_table(index='Month',columns='Year',values=contractlist_r,aggfunc='mean')

    spotplot=px.line(sing5_sppt,width=1000,height=500,title=type+' +'+str(contractlist_r)+'M Rolling Contract Monthly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

elif freq=='Quarterly':
    sing5_sp['Year']=sing5_sp.index.year
    sing5_sp['Quarter']=sing5_sp.index.quarter
    yrsl=st.multiselect('Select Years',options=sing5_sp['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear3')
    sing5_sp=sing5_sp[sing5_sp['Year'].isin(yrsl)]
    sing5_sppt=sing5_sp.pivot_table(index='Quarter',columns='Year',values=contractlist_r,aggfunc='mean')

    spotplot=px.line(sing5_sppt,width=1000,height=500,title=type+' +'+str(contractlist_r)+'M Rolling Contract Quarterly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

st.header(type+' Forward Curve')
sllist2=st.multiselect('Select Contracts',options=sing5_pt1.columns,default=['Spot',m1,m2,m3,m4,m5,m6,m9,m12,m24],key='2')
sing5_fc=sing5_pt1[sllist2]
sing5_fct=sing5_fc.transpose()


tday=tday.date()
lday=lday.date()
l2day=l2day.date()
l3day=l3day.date()
l4day=l4day.date()
lweek=lweek.date()
l2week=l2week.date()
l3week=l3week.date()
lmonth=lmonth.date()
l2month=l2month.date()


sllist3=st.multiselect('Select Dates',options=sing5_fct.columns.date,default=[tday,lday,l2day,lweek,l2week,lmonth,l2month],key='3')
sing5_fctsl=sing5_fct[sllist3]
fctplot=px.line(sing5_fctsl,width=1000,height=500,title=type+' Forward Curve')
fctplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
fctplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(fctplot)

st.header(type+' Time Spread')
st.markdown('#### **----Fixed Contracts**')
tsp1=st.selectbox('Select Contract 1',options=[m1]+list(sing5_pt1.columns))
tsp2=st.selectbox('Select Contract 2',options=[m2]+list(sing5_pt1.columns))

if tsp1!=tsp2:
    sing5_tsp=sing5_pt1[[tsp1,tsp2]]
    sing5_tsp.dropna(inplace=True)
    sing5_tsp['Spread']=sing5_tsp[tsp1]-sing5_tsp[tsp2]
    tspplot=px.line(sing5_tsp[['Spread']],width=1000,height=500,title=type+' Fixed Contract Time Spread: '+str(tsp1)+' minus '+str(tsp2))
    tspplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    tspplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    st.plotly_chart(tspplot)

st.markdown('#### **----Rolling Contracts**')
tsp1_r=st.selectbox('Select Contracts (+n Months)',options=[1]+list(sing5_pt2.columns))
tsp2_r=st.selectbox('Select Contracts (+n Months)',options=[2]+list(sing5_pt2.columns))

if tsp1_r!=tsp2_r:
    sing5_tsp=sing5_pt2[[tsp1_r,tsp2_r]]
    sing5_tsp.dropna(inplace=True)
    sing5_tsp['Spread']=sing5_tsp[tsp1_r]-sing5_tsp[tsp2_r]
    tspplot=px.line(sing5_tsp[['Spread']],width=1000,height=500,title=type+' Rolling Contract Time Spread: +'+str(tsp1_r)+'M minus +'+str(tsp2_r)+'M')
    tspplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    tspplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    st.plotly_chart(tspplot)


freq_r=st.radio('Select Frequency',options=['Weekly','Monthly','Quarterly'],key='freq_r')
if freq_r=='Weekly':
    sing5_tsp['Year']=sing5_tsp.index.year
    sing5_tsp['Week']=sing5_tsp.index.isocalendar().week
    sing5_tsp.loc[sing5_tsp[sing5_tsp.index.date==date(2016,1,2)].index,'Week']=0
    sing5_tsp.loc[sing5_tsp[sing5_tsp.index.date==date(2021,1,2)].index,'Week']=0
    sing5_tsp.loc[sing5_tsp[sing5_tsp.index.date==date(2022,1,1)].index,'Week']=0
    yrsl=st.multiselect('Select Years',options=sing5_tsp['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear11')
    sing5_tsp=sing5_tsp[sing5_tsp['Year'].isin(yrsl)]
    sing5_sppt=sing5_tsp.pivot_table(index='Week',columns='Year',values='Spread',aggfunc='mean')

    spotplot=px.line(sing5_sppt,width=1000,height=500,title=type+' Rolling Contract Time Spread Weekly Seasonality: +'+str(tsp1_r)+'M minus +'+str(tsp2_r)+'M')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

elif freq_r=='Monthly':
    sing5_tsp['Year']=sing5_tsp.index.year
    sing5_tsp['Month']=sing5_tsp.index.month
    yrsl=st.multiselect('Select Years',options=sing5_tsp['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear22')
    sing5_tsp=sing5_tsp[sing5_tsp['Year'].isin(yrsl)]
    sing5_sppt=sing5_tsp.pivot_table(index='Month',columns='Year',values='Spread',aggfunc='mean')

    spotplot=px.line(sing5_sppt,width=1000,height=500,title=type+' Monthly Seasonality of Time Spread +'+str(tsp1_r)+' Months Contract minus +'+str(tsp2_r)+' Months Contract')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

elif freq_r=='Quarterly':
    sing5_tsp['Year']=sing5_tsp.index.year
    sing5_tsp['Quarter']=sing5_tsp.index.quarter
    yrsl=st.multiselect('Select Years',options=sing5_tsp['Year'].unique(),default=np.arange(curryear-4,curryear),key='spotyear33')
    sing5_tsp=sing5_tsp[sing5_tsp['Year'].isin(yrsl)]
    sing5_sppt=sing5_tsp.pivot_table(index='Quarter',columns='Year',values='Spread',aggfunc='mean')

    spotplot=px.line(sing5_sppt,width=1000,height=500,title=type+' Sing 0.5 Quarterly Seasonality of Time Spread +'+str(tsp1_r)+' Months Contract minus +'+str(tsp2_r)+' Months Contract')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)



st.header('Type Spread')


type1=st.selectbox('Select Type 1',options=['SING_0.5','SIN_380','RDM_0.5','RDM35FO'],key='111')
type2=st.selectbox('Select Type 2',options=['RDM_0.5','SING_0.5','SIN_380','RDM35FO'],key='222')


type1_f=bunker_f[bunker_f['Route']==type1]
type1_s=bunker_s[bunker_s['Route']==type1]
type1_f.sort_values(by='Date',ascending=True,inplace=True)

type1_pt1=type1_f.pivot_table(index='Date',columns='Fixed Contract',values='Amount',aggfunc='mean')
type1_pt1.index=pd.to_datetime(type1_pt1.index,dayfirst=True)
type1_pt1.sort_index(inplace=True)

type1_pt2=type1_f.pivot_table(index='Date',columns='Rolling Month Gap',values='Amount',aggfunc='mean')
type1_pt2.sort_index(inplace=True)

type1_s.set_index('Date',inplace=True)
type1_s.sort_index(ascending=True,inplace=True)
type1_s=type1_s[['Amount']]

type1_s.rename(columns={'Amount':'Spot'},inplace=True)
type1_s.drop_duplicates(inplace=True)


type1_pt1=pd.merge(type1_s,type1_pt1,left_index=True,right_index=True,how='outer')
type1_pt2=pd.merge(type1_s,type1_pt2,left_index=True,right_index=True,how='outer')
type1_pt2.rename(columns={'Spot':0},inplace=True)


type2_f=bunker_f[bunker_f['Route']==type2]
type2_s=bunker_s[bunker_s['Route']==type2]
type2_f.sort_values(by='Date',ascending=True,inplace=True)

type2_pt1=type2_f.pivot_table(index='Date',columns='Fixed Contract',values='Amount',aggfunc='mean')
type2_pt1.index=pd.to_datetime(type2_pt1.index,dayfirst=True)
type2_pt1.sort_index(inplace=True)

type2_pt2=type2_f.pivot_table(index='Date',columns='Rolling Month Gap',values='Amount',aggfunc='mean')
type2_pt2.sort_index(inplace=True)

type2_s.set_index('Date',inplace=True)
type2_s.sort_index(ascending=True,inplace=True)
type2_s=type2_s[['Amount']]

type2_s.rename(columns={'Amount':'Spot'},inplace=True)
type2_s.drop_duplicates(inplace=True)


type2_pt1=pd.merge(type2_s,type2_pt1,left_index=True,right_index=True,how='outer')
type2_pt2=pd.merge(type2_s,type2_pt2,left_index=True,right_index=True,how='outer')
type2_pt2.rename(columns={'Spot':0},inplace=True)


ssp_opt=list(set(type1_pt1.columns)&set(type2_pt1.columns))
ssp_opt.sort()

fcssp_multiopt=st.multiselect('Select Contracts for Forward Curve',options=ssp_opt,default=['Spot',m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m15,m18,m21,m24],key='10')
type1_fcssp=type1_pt1[fcssp_multiopt]
type2_fcssp=type2_pt1[fcssp_multiopt]

fcssp_opt=list(set(type1_fcssp.index)&set(type2_fcssp.index))
fcssp_opt.sort(reverse=True)
fcssp=st.selectbox('Select Date for Forward Curve',options=fcssp_opt)
type1_fcssp=type1_fcssp.filter(items=[fcssp],axis=0)
type1_fcssp=type1_fcssp.transpose()
type1_fcssp.columns=[type1]
type2_fcssp=type2_fcssp.filter(items=[fcssp],axis=0)
type2_fcssp=type2_fcssp.transpose()
type2_fcssp.columns=[type2]

cp_fcssp=pd.merge(type1_fcssp,type2_fcssp,how='outer',left_index=True,right_index=True)


fcsspplot=px.line(cp_fcssp,width=1000,height=500,title=type+' Forward Curve Type Spread')
fcsspplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
fcsspplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(fcsspplot)

st.markdown('#### **----Fixed Contracts**')

ssp1=st.multiselect('Choose Contract',options=ssp_opt,default=[m1],key='2011')
ssp_chart1=pd.DataFrame()
for i in ssp1:
    type1_ssp1=type1_pt1[[i]]
    type1_ssp1.columns=[type1]
    type2_ssp1=type2_pt1[[i]]
    type2_ssp1.columns=[type2]
    ssp_mg1=pd.merge(type1_ssp1,type2_ssp1,how='inner',left_index=True,right_index=True)
    ssp_mg1[str(i)+' Type Spread']=ssp_mg1[type1]-ssp_mg1[type2]
    ssp_mg1.dropna(inplace=True)
    ssp_chart1=pd.merge(ssp_chart1,ssp_mg1[[str(i)+' Type Spread']],left_index=True,right_index=True,how='outer')

sspplot1=px.line(ssp_chart1,width=1000,height=500,title='Fixed Contract Type Spread: '+type1+' minus '+type2)
sspplot1.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
sspplot1.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(sspplot1)


#####
st.markdown('#### **----Rolling Contracts**')

type1_pt2.columns=type1_pt2.columns.astype('int64')
type2_pt2.columns=type2_pt2.columns.astype('int64')

rsp_opt=pd.Series(type1_pt2.columns.values)


rsp=st.selectbox('Select Contracts (+n Months)',options=['1']+list(rsp_opt),key='300')

type1_pt2=type1_pt2.add_prefix(type1+'+M',axis=1)
type2_pt2=type2_pt2.add_prefix(type2+'+M',axis=1)

rsp1=type1+'+M'+str(rsp)
rsp2=type2+'+M'+str(rsp)
rsp_sp='+'+str(rsp)+'M Spread'


rsp_chart=pd.merge(type1_pt2[rsp1],type2_pt2[rsp2],left_index=True,right_index=True,how='inner')

rsp_chart[rsp_sp]=rsp_chart[rsp1]-rsp_chart[rsp2]

rspplot=px.line(rsp_chart[rsp_sp],width=1000,height=500,title='+'+str(rsp)+'M Rolling Contract Type Spread: '+str(type1)+' Minus '+str(type2))
rspplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
rspplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(rspplot)


freq_r=st.radio('Select Frequency',options=['Weekly','Monthly','Quarterly'],key='freq_rsp')
if freq_r=='Weekly':
    rsp_chart['Year']=rsp_chart.index.year
    rsp_chart['Week']=rsp_chart.index.isocalendar().week
    rsp_chart.loc[rsp_chart[rsp_chart.index.date==date(2016,1,2)].index,'Week']=0
    rsp_chart.loc[rsp_chart[rsp_chart.index.date==date(2021,1,2)].index,'Week']=0
    rsp_chart.loc[rsp_chart[rsp_chart.index.date==date(2022,1,1)].index,'Week']=0
    yrsl=st.multiselect('Select Years',options=rsp_chart['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear11r')
    rsp_chart=rsp_chart[rsp_chart['Year'].isin(yrsl)]
    p4tc_sppt=rsp_chart.pivot_table(index='Week',columns='Year',values=rsp_sp,aggfunc='mean')

    spotplot=px.line(p4tc_sppt,width=1000,height=500,title='+'+str(rsp)+'M Rolling Contract '+ str(type1)+' Minus '+str(type2)+' Type Spread Weekly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

elif freq_r=='Monthly':
    rsp_chart['Year']=rsp_chart.index.year
    rsp_chart['Month']=rsp_chart.index.month
    yrsl=st.multiselect('Select Years',options=rsp_chart['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear22r')
    rsp_chart=rsp_chart[rsp_chart['Year'].isin(yrsl)]
    p4tc_sppt=rsp_chart.pivot_table(index='Month',columns='Year',values=rsp_sp,aggfunc='mean')

    spotplot=px.line(p4tc_sppt,width=1000,height=500,title='+'+str(rsp)+'M Rolling Contract '+ str(type1)+' Minus '+str(type2)+' Type Spread Monthly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

elif freq_r=='Quarterly':
    rsp_chart['Year']=rsp_chart.index.year
    rsp_chart['Quarter']=rsp_chart.index.quarter
    yrsl=st.multiselect('Select Years',options=rsp_chart['Year'].unique(),default=np.arange(curryear-4,curryear+1),key='spotyear33r')
    rsp_chart=rsp_chart[rsp_chart['Year'].isin(yrsl)]
    p4tc_sppt=rsp_chart.pivot_table(index='Quarter',columns='Year',values=rsp_sp,aggfunc='mean')

    spotplot=px.line(p4tc_sppt,width=1000,height=500,title='+'+str(rsp)+'M Rolling Contract '+ str(type1)+' Minus '+str(type2)+' Type Spread Quarterly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    st.plotly_chart(spotplot)

