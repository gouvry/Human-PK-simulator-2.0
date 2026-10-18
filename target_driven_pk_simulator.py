import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title='Target-driven Human PK Simulator', layout='wide')
st.title('Target-driven Human PK Simulator')
st.caption('The exposure target is the input; the required dose is calculated and used to generate the PK profile.')

with st.sidebar:
    st.header('Human PK inputs')
    cl_ml_min_kg = st.number_input('Plasma clearance (mL/min/kg)', min_value=0.001, value=5.0, step=0.1)
    vd_l_kg = st.number_input('Volume of distribution (L/kg)', min_value=0.001, value=1.0, step=0.1)
    f_pct = st.number_input('Bioavailability (%)', min_value=0.1, max_value=100.0, value=70.0, step=1.0)
    ka = st.number_input('ka (h^-1)', min_value=0.001, value=1.0, step=0.1)
    mw = st.number_input('Molecular weight (g/mol)', min_value=1.0, value=450.0, step=1.0)
    fup = st.number_input('Fraction unbound in plasma (fu,p)', min_value=0.000001, max_value=1.0, value=0.10, step=0.01, format='%.4f')
    kpuu = st.number_input('Kp,uu (unbound brain / unbound plasma)', min_value=0.000001, value=1.0, step=0.1, format='%.3f')
    bw = st.number_input('Body weight (kg)', min_value=1.0, value=70.0, step=1.0)
    regimen = st.selectbox('Regimen', ['QD', 'BID'])
    days = st.slider('Simulation days', 1, 28, 7)
    ic50 = st.number_input('IC50 (nM)', min_value=0.001, value=100.0, step=10.0)

st.subheader('Dose target')
target_site = st.radio('Target site', ['Plasma', 'Brain'], horizontal=True)
target_mode = st.radio('Target type', ['Ctrough concentration', 'Average inhibition'], horizontal=True)
if target_mode == 'Ctrough concentration':
    concentration_target = st.selectbox('Ctrough target', ['IC50', 'IC90', 'Other'])
    if concentration_target == 'Other':
        target_unit = st.selectbox('Target units', ['nM', 'ng/mL'])
        target_value = st.number_input(f'Target free {target_site.lower()} Ctrough ({target_unit})', min_value=0.001, value=100.0)
    elif concentration_target == 'IC50':
        target_unit, target_value = 'nM', ic50
        st.info(f'Target free {target_site.lower()} Ctrough = IC50 = {target_value:g} nM')
    else:
        target_unit, target_value = 'nM', 9.0 * ic50
        st.info(f'For Hill slope 1, IC90 = 9 × IC50 = {target_value:g} nM')
else:
    target_inhibition = st.number_input('Target average inhibition (%)', min_value=0.1, max_value=99.9, value=80.0)

cl = cl_ml_min_kg * 60.0 / 1000.0 * bw
vd = vd_l_kg * bw
kel = cl / vd
f = f_pct / 100.0
tau = 24.0 if regimen == 'QD' else 12.0
site_factor = fup if target_site == 'Plasma' else fup * kpuu
if abs(ka-kel) < 1e-8:
    st.error('ka is effectively equal to kel; choose a slightly different value.')
    st.stop()

def single_c(t, dose_mg):
    t = np.asarray(t, dtype=float)
    return f*dose_mg*ka/(vd*(ka-kel))*(np.exp(-kel*t)-np.exp(-ka*t))

def profile(dose_mg, total_h, dt=0.02):
    t = np.arange(0.0, total_h+dt/2, dt)
    c = np.zeros_like(t)
    for td in np.arange(0.0, total_h+1e-9, tau):
        mask = t >= td
        c[mask] += single_c(t[mask]-td, dose_mg)
    return t, c

def mgL_to_nm(c): return c*1e6/mw

def metrics(t, c, start, end):
    mask=(t>=start)&(t<=end+1e-9)
    tw,cw=t[mask],c[mask]
    auc=np.trapezoid(cw,tw)
    doses=np.arange(math.ceil(start/tau)*tau,end+1e-9,tau)
    predose=[]
    for td in doses:
        if td <= start+1e-9: continue
        idx=np.searchsorted(t,td)-1
        if idx>=0: predose.append(c[idx])
    return {'Cmax':float(np.max(cw)), 'Ctrough':float(min(predose)) if predose else float(cw[-1]), 'Cavg':auc/(end-start), 'AUC24':auc}

def avg_inhibition(t,c,start,end,factor):
    mask=(t>=start)&(t<=end+1e-9)
    tw,cw=t[mask],c[mask]
    free_nm=mgL_to_nm(cw)*factor
    return float(np.trapezoid(free_nm/(free_nm+ic50),tw)/(end-start))

total_h=max(24.0,days*24.0)
last_start=total_h-24.0

def target_result(dose):
    t,c=profile(dose,total_h)
    m=metrics(t,c,last_start,total_h)
    if target_mode == 'Average inhibition':
        return 100*avg_inhibition(t,c,last_start,total_h,site_factor)
    free_trough=m['Ctrough']*site_factor
    return mgL_to_nm(free_trough) if target_unit=='nM' else free_trough*1000.0

try:
    if target_mode == 'Ctrough concentration':
        per_mg=target_result(1.0)
        required_dose=target_value/per_mg
    else:
        required_dose=brentq(lambda d:target_result(d)-target_inhibition,1e-6,1e7)
except Exception:
    st.error('The requested target could not be solved within the model range.')
    st.stop()

st.success(f'Calculated dose per administration: {required_dose:.1f} mg ({regimen})')
t,c=profile(required_dose,total_h)
first=metrics(t,c,0.0,24.0)
last=metrics(t,c,last_start,total_h)

st.subheader('Concentration-time profile')
a,b=st.columns(2)
with a: unit=st.radio('Concentration units',['nM','ng/mL'],horizontal=True)
with b: log_y=st.toggle('Logarithmic y-axis',value=False)
show_total=st.checkbox('Total plasma',True)
show_free_plasma=st.checkbox('Free plasma',True)
show_free_brain=st.checkbox('Free brain',True)
base=mgL_to_nm(c) if unit=='nM' else c*1000.0
fig=go.Figure()
if show_total: fig.add_trace(go.Scatter(x=t,y=base,mode='lines',name='Total plasma'))
if show_free_plasma: fig.add_trace(go.Scatter(x=t,y=base*fup,mode='lines',name='Free plasma'))
if show_free_brain: fig.add_trace(go.Scatter(x=t,y=base*fup*kpuu,mode='lines',name='Free brain'))
fig.update_layout(xaxis_title='Time (hours)',yaxis_title=f'Concentration ({unit})',hovermode='x unified',yaxis_type='log' if log_y else 'linear')
st.plotly_chart(fig,use_container_width=True)

def show_metrics(m,label,start,end):
    conv=(lambda x:mgL_to_nm(x)) if unit=='nM' else (lambda x:x*1000.0)
    auc_conv=mgL_to_nm(m['AUC24']) if unit=='nM' else m['AUC24']*1000.0
    st.markdown(f'**{label}**')
    profiles=[]
    if show_total: profiles.append(('Total plasma',1.0))
    if show_free_plasma: profiles.append(('Free plasma',fup))
    if show_free_brain: profiles.append(('Free brain',fup*kpuu))
    for name,factor in profiles:
        st.caption(name)
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Cmax',f"{conv(m['Cmax']*factor):,.0f} {unit}")
        c2.metric('Ctrough',f"{conv(m['Ctrough']*factor):,.0f} {unit}")
        c3.metric('Cavg',f"{conv(m['Cavg']*factor):,.0f} {unit}")
        c4.metric('AUC0-24',f"{auc_conv*factor:,.0f} {unit}·h")
    p,b=st.columns(2)
    p.metric('Average inhibition, free plasma',f'{100*avg_inhibition(t,c,start,end,fup):.1f}%')
    b.metric('Average inhibition, free brain',f'{100*avg_inhibition(t,c,start,end,fup*kpuu):.1f}%')

show_metrics(first,'First 24 hours',0.0,24.0)
show_metrics(last,'Final complete 24-hour window',last_start,total_h)

with st.expander('Model assumptions'):
    st.write('One compartment; first-order oral absorption and elimination; linear dose-proportional PK; Hill slope = 1.')
    st.write('Free plasma = total plasma × fu,p. Free brain = free plasma × Kp,uu. IC90 = 9 × IC50 for Hill slope 1.')
    st.write('The dose is calculated against the selected free-concentration target in the final complete 24-hour window displayed.')
