import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import io
import qrcode
from astroquery.jplhorizons import Horizons
import plotly.graph_objects as go
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
# ════════════════════════════════════════════════════════════════════
#  SAYFA AYARLARI VE YAN MENÜ (SIDEBAR)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Gök Mekaniği Raporlayıcı", layout="centered", page_icon="🪐")

st.sidebar.title("Proje Hakkında")
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Tüm Hakları Saklıdır.\n"
                   "Yiğit Efe Devecioğlu\n"
                   "deveciogluyigitefe@gmail.com\n\n"
                   "Bu araç, yörünge dinamikleri hesaplamalarını otomatize "
                   "etmek amacıyla geliştirilmiş açık kaynaklı bir projedir.")

# ════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR (Önceki kodun aynısı)
# ════════════════════════════════════════════════════════════════════
def jd_to_date(jd):
    jd = jd + 0.5
    Z  = int(jd); F = jd - Z
    if Z >= 2299161:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - alpha // 4
    else:
        A = Z
    B = A + 1524; C = int((B - 122.1) / 365.25)
    D = int(365.25 * C); E_c = int((B - D) / 30.6001)
    day   = B - D - int(30.6001 * E_c) + F
    month = E_c - 1 if E_c < 14 else E_c - 13
    year  = C - 4716 if month > 2 else C - 4715
    saat = (day - int(day)) * 24
    dk = (saat - int(saat)) * 60
    return f"{year}-{month:02d}-{int(day):02d} {int(saat):02d}:{int(dk):02d}"

def kepler_cozucu(M_array, e, tol=1e-6, max_iter=100):
    M_mod = np.mod(M_array, 2 * np.pi)
    E = np.where(M_mod < np.pi, M_mod + e / 2, M_mod - e / 2)
    for _ in range(max_iter):
        delta = (E - e * np.sin(E) - M_mod) / (1 - e * np.cos(E))
        E -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return E

def hesapla(t, a, e, P, tau):
    M  = np.mod((2 * np.pi / P) * (t - tau), 2 * np.pi)
    E  = kepler_cozucu(M, e)
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    nu = np.mod(nu, 2 * np.pi)
    r  = a * (1 - e * np.cos(E))
    return M, E, nu, r, r*np.cos(nu), r*np.sin(nu)

def gezegen_ciz(ax, a_g, e_g, isim, renk):
    E_g = np.linspace(0, 2*np.pi, 300)
    ax.plot(a_g*(np.cos(E_g)-e_g), a_g*np.sqrt(1-e_g**2)*np.sin(E_g),
            color=renk, linestyle='--', linewidth=0.65, alpha=0.6, label=isim)

def anlamli_noktalar(a, e, P, tau):
    def t_from_M(M_h, d=0): return tau + (M_h + 2*np.pi*d)*P/(2*np.pi)
    def E_for_nu(nu_h):
        t = np.tan(nu_h/2) / np.sqrt((1+e)/(1-e))
        return np.mod(2*np.arctan(t), 2*np.pi)
    def M_from_E(E): return np.mod(E - e*np.sin(E), 2*np.pi)
    E2 = E_for_nu(np.pi/2);  M2 = M_from_E(E2); r2 = a*(1-e*np.cos(E2))
    E4 = E_for_nu(3*np.pi/2);M4 = M_from_E(E4); r4 = a*(1-e*np.cos(E4))
    return [
        (t_from_M(np.pi,-1), np.pi, np.pi,  np.pi,       a*(1+e), "Önceki Enöte"),
        (t_from_M(0,    0),  0,    0,       0,            a*(1-e), "Enberi (ν=0°)"),
        (t_from_M(M2,   0),  M2,   E2,      np.pi/2,     r2,      "ν = 90°"),
        (t_from_M(np.pi,0),  np.pi,np.pi,   np.pi,       a*(1+e), "Enöte (ν=180°)"),
        (t_from_M(M4,   0),  M4,   E4,      3*np.pi/2,   r4,      "ν = 270°"),
        (t_from_M(M2,   1),  M2,   E2,      np.pi/2,     r2,      "Sonraki ν=90°"),
    ]
def uzay_3d_donusum(x, y, i_deg, W_deg, w_deg):
    i_rad, W_rad, w_rad = np.radians(i_deg), np.radians(W_deg), np.radians(w_deg)
    
    X = x * (np.cos(W_rad)*np.cos(w_rad) - np.sin(W_rad)*np.sin(w_rad)*np.cos(i_rad)) + \
        y * (-np.cos(W_rad)*np.sin(w_rad) - np.sin(W_rad)*np.cos(w_rad)*np.cos(i_rad))
    Y = x * (np.sin(W_rad)*np.cos(w_rad) + np.cos(W_rad)*np.sin(w_rad)*np.cos(i_rad)) + \
        y * (-np.sin(W_rad)*np.sin(w_rad) + np.cos(W_rad)*np.cos(w_rad)*np.cos(i_rad))
    Z = x * (np.sin(w_rad)*np.sin(i_rad)) + \
        y * (np.cos(w_rad)*np.sin(i_rad))
    return X, Y, Z
def eleman_to_vektor(a, e, i_deg, W_deg, w_deg, nu_deg, mu):
    # PDF Sayfa 7-8: Elemanlardan Durum Vektörüne Dönüşüm
    i, W, w, nu = np.radians(i_deg), np.radians(W_deg), np.radians(w_deg), np.radians(nu_deg)
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))
    
    x_bar = r_mag * np.cos(nu)
    y_bar = r_mag * np.sin(nu)
    
    X, Y, Z = uzay_3d_donusum(x_bar, y_bar, i_deg, W_deg, w_deg)
    
    h = np.sqrt(mu * p)
    vx_bar = -(mu / h) * np.sin(nu)
    vy_bar = (mu / h) * (e + np.cos(nu))
    
    VX, VY, VZ = uzay_3d_donusum(vx_bar, vy_bar, i_deg, W_deg, w_deg)
    return np.array([X, Y, Z]), np.array([VX, VY, VZ]), r_mag

def vektor_to_eleman(R, V, mu):
    # PDF Sayfa 2-5: Durum Vektöründen Elemanlara Dönüşüm
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = np.dot(R, V) / r
    
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    i = np.arccos(H[2] / h)
    
    K = np.array([0, 0, 1])
    N_vec = np.cross(K, H)
    n = np.linalg.norm(N_vec)
    
    W = np.arccos(N_vec[0] / n) if n != 0 else 0.0
    if n != 0 and N_vec[1] < 0: W = 2 * np.pi - W
        
    E_vec = (1/mu) * ((v**2 - mu/r)*R - r*vr*V)
    e = np.linalg.norm(E_vec)
    
    w = 0.0
    if n != 0 and e != 0:
        w = np.arccos(np.clip(np.dot(N_vec, E_vec) / (n * e), -1.0, 1.0))
        if E_vec[2] < 0: w = 2 * np.pi - w
        
    nu = 0.0
    if e != 0:
        nu = np.arccos(np.clip(np.dot(E_vec, R) / (e * r), -1.0, 1.0))
        if vr < 0: nu = 2 * np.pi - nu
        
    a = (h**2 / mu) * (1 / (1 - e**2)) if abs(e - 1.0) > 1e-6 else np.inf
    
    return a, e, np.degrees(i), np.degrees(W), np.degrees(w), np.degrees(nu)
def su_anki_jd():
    # Şu anki UTC zamanını alıp Julian Day'e çeviren fonksiyon
    simdi = datetime.now(timezone.utc)
    yil, ay, gun = simdi.year, simdi.month, simdi.day
    saat, dk, sn = simdi.hour, simdi.minute, simdi.second
    
    if ay <= 2:
        yil -= 1
        ay += 12
    
    A = yil // 100
    B = 2 - A + (A // 4)
    jd_tam = int(365.25 * (yil + 4716)) + int(30.6001 * (ay + 1)) + gun + B - 1524.5
    jd_kesir = (saat + dk / 60.0 + sn / 3600.0) / 24.0
    return jd_tam + jd_kesir

def plotly_3d_ciz(a, e, P, tau, i, W, w, cisim_ismi):
    t_g = np.linspace(tau, tau+P, 1000)
    _, _, _, r_g, x_g, y_g = hesapla(t_g, a, e, P, tau)
    X_g, Y_g, Z_g = uzay_3d_donusum(x_g, y_g, i, W, w)
    
    idx_enberi, idx_enote = np.argmin(r_g), np.argmax(r_g)
    
    jd_su_an = su_anki_jd()
    _, _, _, _, x_suan, y_suan = hesapla(jd_su_an, a, e, P, tau)
    X_suan, Y_suan, Z_suan = uzay_3d_donusum(x_suan, y_suan, i, W, w)
    if isinstance(X_suan, np.ndarray):
        X_suan, Y_suan, Z_suan = X_suan[0], Y_suan[0], Z_suan[0]
        
    fig = go.Figure()
    
    # 1. Güneş ve Ana Cisim
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=12, color='#f39c12', line=dict(color='white', width=1.5)), name='Güneş', hoverinfo='name'))
    fig.add_trace(go.Scatter3d(x=X_g, y=Y_g, z=Z_g, mode='lines', line=dict(color='#1a2940', width=5), name=f'{cisim_ismi} Yörüngesi'))
    fig.add_trace(go.Scatter3d(x=[X_suan], y=[Y_suan], z=[Z_suan], mode='markers', marker=dict(size=8, color='#e74c3c', line=dict(color='white', width=1)), name=f'{cisim_ismi} (Şu An)'))
    
    # 2. Vektörler
    fig.add_trace(go.Scatter3d(x=[0, X_g[idx_enberi]], y=[0, Y_g[idx_enberi]], z=[0, Z_g[idx_enberi]], mode='lines+text', line=dict(color='#1e8449', width=2, dash='dash'), text=['', f'Enberi: {r_g[idx_enberi]:.2f} AB'], textposition='top center', textfont=dict(size=10, color='#1e8449'), name='Enberi Vektörü'))
    fig.add_trace(go.Scatter3d(x=[0, X_g[idx_enote]], y=[0, Y_g[idx_enote]], z=[0, Z_g[idx_enote]], mode='lines+text', line=dict(color='#7d3c98', width=2, dash='dash'), text=['', f'Enöte: {r_g[idx_enote]:.2f} AB'], textposition='top center', textfont=dict(size=10, color='#7d3c98'), name='Enöte Vektörü'))

    # 3. Dünya (Yörünge ve Şu Anki Konum)
    _, _, _, _, x_e, y_e = hesapla(np.linspace(0, 365.25, 300), 1.0, 0.0167, 365.25, 0)
    Xe, Ye, Ze = uzay_3d_donusum(x_e, y_e, 0.0, 0.0, 102.9)
    fig.add_trace(go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=dict(color='#2980b9', width=2, dash='dot'), name='Dünya Yörüngesi', hoverinfo='name'))
    
    _, _, _, _, x_e_suan, y_e_suan = hesapla(jd_su_an, 1.0, 0.0167, 365.25, 0)
    Xe_suan, Ye_suan, Ze_suan = uzay_3d_donusum(x_e_suan, y_e_suan, 0.0, 0.0, 102.9)
    if isinstance(Xe_suan, np.ndarray): Xe_suan, Ye_suan, Ze_suan = Xe_suan[0], Ye_suan[0], Ze_suan[0]
    fig.add_trace(go.Scatter3d(x=[Xe_suan], y=[Ye_suan], z=[Ze_suan], mode='markers', marker=dict(size=5, color='#3498db', line=dict(color='white', width=1)), name='Dünya (Şu An)'))

    # 4. Jüpiter (Yörünge ve Şu Anki Konum)
    _, _, _, _, x_j, y_j = hesapla(np.linspace(0, 4332.59, 500), 5.204, 0.0489, 4332.59, 0)
    Xj, Yj, Zj = uzay_3d_donusum(x_j, y_j, 1.30, 100.5, 273.8)
    fig.add_trace(go.Scatter3d(x=Xj, y=Yj, z=Zj, mode='lines', line=dict(color='#c0392b', width=2, dash='dot'), name='Jüpiter Yörüngesi', hoverinfo='name'))
    
    _, _, _, _, x_j_suan, y_j_suan = hesapla(jd_su_an, 5.204, 0.0489, 4332.59, 0)
    Xj_suan, Yj_suan, Zj_suan = uzay_3d_donusum(x_j_suan, y_j_suan, 1.30, 100.5, 273.8)
    if isinstance(Xj_suan, np.ndarray): Xj_suan, Yj_suan, Zj_suan = Xj_suan[0], Yj_suan[0], Zj_suan[0]
    fig.add_trace(go.Scatter3d(x=[Xj_suan], y=[Yj_suan], z=[Zj_suan], mode='markers', marker=dict(size=7, color='#e67e22', line=dict(color='white', width=1)), name='Jüpiter (Şu An)'))

    # 5. Ekliptik Düzlem
    xy_limit = max(abs(X_g).max(), abs(Y_g).max(), 5.5) * 1.05
    grid_val = np.linspace(-xy_limit, xy_limit, 2)
    xg, yg = np.meshgrid(grid_val, grid_val)
    fig.add_trace(go.Surface(x=xg, y=yg, z=np.zeros_like(xg), opacity=0.08, showscale=False, colorscale='Greys', name='Ekliptik Düzlem', hoverinfo='skip'))
    
    fig.update_layout(
        scene=dict(xaxis=dict(title='X (AB)'), yaxis=dict(title='Y (AB)'), zaxis=dict(title='Z (AB)'), aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=80),
        title=dict(text=f"<b>{cisim_ismi} - Şu Anki Konum Modeli</b>", x=0.5, y=0.95, font=dict(size=16)),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.2, 
            xanchor="center", 
            x=0.5, 
            bgcolor="rgba(0,0,0,0)", 
            borderwidth=0
        )
    )
    return fig
def plotly_3d_ciz_jenerik(a, e, i, W, w, nu, mu, cisim_ismi, merkez_ismi="Dünya"):
    # Dünya veya Özel Merkezli Jenerik (Akademik) Simülatör
    nu_array = np.linspace(0, 2*np.pi, 500)
    p = a * (1 - e**2)
    r_array = p / (1 + e * np.cos(nu_array))
    
    # 2D Yörünge -> 3D Dönüşümü
    x_bar = r_array * np.cos(nu_array)
    y_bar = r_array * np.sin(nu_array)
    X_g, Y_g, Z_g = uzay_3d_donusum(x_bar, y_bar, i, W, w)
    
    # Şu anki Konum (nu açısında)
    r_suan = p / (1 + e * np.cos(np.radians(nu)))
    X_suan, Y_suan, Z_suan = uzay_3d_donusum(r_suan * np.cos(np.radians(nu)), r_suan * np.sin(np.radians(nu)), i, W, w)
    
    # Akademik Yayların Geometrisi
    R_arc = a * 0.35
    
    # 1. Ω Yayı (Referans düzlemde)
    th_W = np.linspace(0, np.radians(W), 50)
    X_W, Y_W, Z_W = R_arc * np.cos(th_W), R_arc * np.sin(th_W), np.zeros_like(th_W)
    Nx, Ny = X_W[-1], Y_W[-1]
    
    # 2. ω Yayı (Yörünge düzleminde, düğümden enberiye)
    th_w = np.linspace(0, np.radians(w), 50)
    X_w, Y_w, Z_w = uzay_3d_donusum(R_arc * np.cos(th_w), R_arc * np.sin(th_w), i, W, 0)
    
    # 3. i Yayı (Düğüm çizgisine tam 90 derece dik konumda)
    x_i_2d = R_arc * 1.2 * np.cos(np.pi/2)
    y_i_2d = R_arc * 1.2 * np.sin(np.pi/2)
    X_i, Y_i, Z_i = [], [], []
    for i_temp in np.linspace(0, i, 50):
        xt, yt, zt = uzay_3d_donusum(x_i_2d, y_i_2d, i_temp, W, 0)
        X_i.append(xt); Y_i.append(yt); Z_i.append(zt)

    # 4. ν (Gerçek Anomali) Yayı
    th_nu = np.linspace(0, np.radians(nu), 50)
    X_nu, Y_nu, Z_nu = uzay_3d_donusum(R_arc * 0.8 * np.cos(th_nu), R_arc * 0.8 * np.sin(th_nu), i, W, w)

    # Durum Vektörleri ve Enberi Doğrultusu Hesapları
    R_vec, V_vec, _ = eleman_to_vektor(a, e, i, W, w, nu, mu)
    V_scale = (a * 0.4) / np.linalg.norm(V_vec) # Hız vektörünü yörünge ölçeğine görsel olarak uydurur
    X_enb, Y_enb, Z_enb = uzay_3d_donusum(a*(1-e), 0, i, W, w)

    fig = go.Figure()
    
    merkez_renk = '#3498db' if merkez_ismi == "Dünya" else '#9b59b6'
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=14, color=merkez_renk, line=dict(color='white', width=1)), name=merkez_ismi))
    
    # Yörünge ve Saydam "Cam" Düzlem
    fig.add_trace(go.Scatter3d(x=X_g, y=Y_g, z=Z_g, mode='lines', line=dict(color='#1a2940', width=4), name=f'{cisim_ismi} Yörüngesi'))
    fig.add_trace(go.Mesh3d(x=[0]+list(X_g), y=[0]+list(Y_g), z=[0]+list(Z_g), i=[0]*(len(X_g)-1), j=list(range(1, len(X_g))), k=list(range(2, len(X_g)+1)), color='#3498db', opacity=0.15, name='Yörünge Düzlemi', hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[X_suan], y=[Y_suan], z=[Z_suan], mode='markers', marker=dict(size=7, color='#e74c3c'), name='Cismin Konumu'))
    
    # Referans Eksenleri ve Temel Vektörler
    fig.add_trace(go.Scatter3d(x=[0, max(X_g)*1.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='gray', width=2), name='X Ekseni (Koç Noktası)'))
    fig.add_trace(go.Scatter3d(x=[0, Nx*3], y=[0, Ny*3], z=[0, 0], mode='lines', line=dict(color='#8e44ad', width=2, dash='dashdot'), name='Düğüm Çizgisi'))
    
    # PDF Sayfa 5'teki Vektörlerin Eklenmesi (Silik/Kesikli)
    fig.add_trace(go.Scatter3d(x=[0, X_enb], y=[0, Y_enb], z=[0, Z_enb], mode='lines', line=dict(color='#f39c12', width=2, dash='dot'), name='Enberi Doğrultusu (e)'))
    fig.add_trace(go.Scatter3d(x=[0, R_vec[0]], y=[0, R_vec[1]], z=[0, R_vec[2]], mode='lines', line=dict(color='#3498db', width=3, dash='dash'), name='Konum Vektörü (r)'))
    fig.add_trace(go.Scatter3d(x=[R_vec[0], R_vec[0] + V_vec[0]*V_scale], y=[R_vec[1], R_vec[1] + V_vec[1]*V_scale], z=[R_vec[2], R_vec[2] + V_vec[2]*V_scale], mode='lines', line=dict(color='#e74c3c', width=4), name='Hız Vektörü (v)'))
    
    # Hover Metinleri ile Yaylar (Kalın Çizgiler)
    h_W = "<b>Çıkış Düğümü Boylamı (Ω):</b><br>Referans X ekseninden düğüm çizgisine olan açıdır."
    fig.add_trace(go.Scatter3d(x=X_W, y=Y_W, z=Z_W, mode='lines', line=dict(color='#2ecc71', width=5), name='Ω', hovertemplate=h_W))
    
    h_w = "<b>Enberi Argümanı (ω):</b><br>Düğüm çizgisinden Enberi noktasına olan açıdır."
    fig.add_trace(go.Scatter3d(x=X_w, y=Y_w, z=Z_w, mode='lines', line=dict(color='#e67e22', width=5), name='ω', hovertemplate=h_w))
    
    h_i = "<b>Eğiklik (i):</b><br>Yörünge düzleminin referans düzlemle yaptığı açıdır.<br>Düğüme dik ölçülür."
    fig.add_trace(go.Scatter3d(x=X_i, y=Y_i, z=Z_i, mode='lines', line=dict(color='#3498db', width=5), name='i', hovertemplate=h_i))

    h_nu = "<b>Gerçek Anomali (ν):</b><br>Enberi noktasından cismin anlık konumuna olan açıdır."
    fig.add_trace(go.Scatter3d(x=X_nu, y=Y_nu, z=Z_nu, mode='lines', line=dict(color='#9b59b6', width=5), name='ν', hovertemplate=h_nu))

    # 3D Uzayda Yüzen Matematiksel Etiketler (Semboller)
    fig.add_trace(go.Scatter3d(x=[X_W[len(X_W)//2]], y=[Y_W[len(Y_W)//2]], z=[Z_W[len(Z_W)//2]], mode='text', text=['Ω'], textfont=dict(size=18, color='#2ecc71'), textposition='bottom center', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[X_w[len(X_w)//2]], y=[Y_w[len(Y_w)//2]], z=[Z_w[len(Z_w)//2]], mode='text', text=['ω'], textfont=dict(size=18, color='#e67e22'), textposition='top center', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[X_i[len(X_i)//2]], y=[Y_i[len(Y_i)//2]], z=[Z_i[len(Z_i)//2]], mode='text', text=['i'], textfont=dict(size=18, color='#3498db'), textposition='middle right', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[X_nu[len(X_nu)//2]], y=[Y_nu[len(Y_nu)//2]], z=[Z_nu[len(Z_nu)//2]], mode='text', text=['ν'], textfont=dict(size=18, color='#9b59b6'), textposition='bottom center', showlegend=False, hoverinfo='skip'))

    fig.update_layout(scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=40),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    return fig
def plotly_3d_ciz_jenerik(a, e, i, W, w, nu, mu, cisim_ismi, merkez_ismi="Dünya"):
    nu_array = np.linspace(0, 2*np.pi, 500)
    p = a * (1 - e**2)
    r_array = p / (1 + e * np.cos(nu_array))
    
    x_bar = r_array * np.cos(nu_array)
    y_bar = r_array * np.sin(nu_array)
    X_g, Y_g, Z_g = uzay_3d_donusum(x_bar, y_bar, i, W, w)
    
    r_suan = p / (1 + e * np.cos(np.radians(nu)))
    X_suan, Y_suan, Z_suan = uzay_3d_donusum(r_suan * np.cos(np.radians(nu)), r_suan * np.sin(np.radians(nu)), i, W, w)
    
    R_arc = a * 0.35
    
    th_W = np.linspace(0, np.radians(W), 50)
    X_W, Y_W, Z_W = R_arc * np.cos(th_W), R_arc * np.sin(th_W), np.zeros_like(th_W)
    Nx, Ny = X_W[-1], Y_W[-1]
    
    th_w = np.linspace(0, np.radians(w), 50)
    X_w, Y_w, Z_w = uzay_3d_donusum(R_arc * np.cos(th_w), R_arc * np.sin(th_w), i, W, 0)
    
    x_i_2d = R_arc * 1.2 * np.cos(np.pi/2)
    y_i_2d = R_arc * 1.2 * np.sin(np.pi/2)
    X_i, Y_i, Z_i = [], [], []
    for i_temp in np.linspace(0, i, 50):
        xt, yt, zt = uzay_3d_donusum(x_i_2d, y_i_2d, i_temp, W, 0)
        X_i.append(xt); Y_i.append(yt); Z_i.append(zt)
        
    th_nu = np.linspace(0, np.radians(nu), 50)
    X_nu, Y_nu, Z_nu = uzay_3d_donusum(R_arc * 0.8 * np.cos(th_nu), R_arc * 0.8 * np.sin(th_nu), i, W, w)

    R_vec, V_vec, _ = eleman_to_vektor(a, e, i, W, w, nu, mu)
    V_scale = (a * 0.4) / np.linalg.norm(V_vec)
    X_enb, Y_enb, Z_enb = uzay_3d_donusum(a*(1-e), 0, i, W, w)

    fig = go.Figure()
    
    # RENK PALETİ
    renk_merkez = '#3498db' if merkez_ismi == "Dünya" else '#9b59b6'
    renk_yorunge = '#0b192c'
    renk_r = '#e0e0e0'
    renk_v = '#e74c3c'
    renk_i = '#00ffff'
    renk_W = '#2ecc71'
    renk_w = '#e67e22'
    renk_nu = '#9b59b6'
    
    # Merkez Cisim
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=14, color=renk_merkez, line=dict(color='white', width=1)), name=merkez_ismi))
    
    # EKVATOR CAM ZEMİNİ
    xy_limit = a * 1.5
    grid_val = np.linspace(-xy_limit, xy_limit, 2)
    xg, yg = np.meshgrid(grid_val, grid_val)
    fig.add_trace(go.Surface(x=xg, y=yg, z=np.zeros_like(xg), opacity=0.04, showscale=False, colorscale=[[0, '#bdc3c7'], [1, '#bdc3c7']], name='Ekvator Camı', hoverinfo='skip'))
    
    # YÖRÜNGE MANTIĞI (Katmanlı Sistem)
    # 1. Daimi Kesikli Çizgi (Menüde yok, hep görünür) - KALINLIK 4 YAPILDI
    fig.add_trace(go.Scatter3d(x=X_g, y=Y_g, z=Z_g, mode='lines', line=dict(color=renk_yorunge, width=4, dash='dash'), showlegend=False, hoverinfo='skip'))
    # 2. Açılıp Kapanabilen Kalın Çizgi ve Cam Düzlem (Birbirine bağlı)
    fig.add_trace(go.Scatter3d(x=X_g, y=Y_g, z=Z_g, mode='lines', line=dict(color=renk_yorunge, width=5), name=f'{cisim_ismi} Düzlemi', legendgroup='yorunge_cam'))
    fig.add_trace(go.Mesh3d(x=[0]+list(X_g), y=[0]+list(Y_g), z=[0]+list(Z_g), i=[0]*(len(X_g)-1), j=list(range(1, len(X_g))), k=list(range(2, len(X_g)+1)), color='#2980b9', opacity=0.25, name='Cam Yüzey', legendgroup='yorunge_cam', showlegend=False, hoverinfo='skip'))
    
    # Vektörler
    fig.add_trace(go.Scatter3d(x=[0, X_enb], y=[0, Y_enb], z=[0, Z_enb], mode='lines', line=dict(color='#f39c12', width=2, dash='dot'), name='Enberi Doğrultusu'))
    fig.add_trace(go.Scatter3d(x=[X_enb], y=[Y_enb], z=[Z_enb], mode='text', text=['Π'], textfont=dict(size=14, color='#f39c12'), textposition='top center', showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter3d(x=[0, R_vec[0]], y=[0, R_vec[1]], z=[0, R_vec[2]], mode='lines', line=dict(color=renk_r, width=3, dash='dash'), name='Konum Vektörü (r)'))
    fig.add_trace(go.Scatter3d(x=[R_vec[0]], y=[R_vec[1]], z=[R_vec[2]], mode='text', text=['r'], textfont=dict(size=14, color=renk_r), textposition='bottom right', showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter3d(x=[R_vec[0], R_vec[0] + V_vec[0]*V_scale], y=[R_vec[1], R_vec[1] + V_vec[1]*V_scale], z=[R_vec[2], R_vec[2] + V_vec[2]*V_scale], mode='lines', line=dict(color=renk_v, width=4), name='Hız Vektörü (v)'))
    fig.add_trace(go.Scatter3d(x=[R_vec[0] + V_vec[0]*V_scale], y=[R_vec[1] + V_vec[1]*V_scale], z=[R_vec[2] + V_vec[2]*V_scale], mode='text', text=['V'], textfont=dict(size=14, color=renk_v), textposition='top right', showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter3d(x=[X_suan], y=[Y_suan], z=[Z_suan], mode='markers', marker=dict(size=7, color=renk_v), name='Cismin Konumu', hoverinfo='skip'))
    
    # Eksenler
    fig.add_trace(go.Scatter3d(x=[0, xy_limit], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='gray', width=2), name='X Ekseni (Koç Noktası)'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, xy_limit], z=[0, 0], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='Y Ekseni (Aç/Kapat)', visible='legendonly'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, xy_limit], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='Z Ekseni (Aç/Kapat)', visible='legendonly'))
    
    fig.add_trace(go.Scatter3d(x=[0, Nx*3], y=[0, Ny*3], z=[0, 0], mode='lines', line=dict(color='#8e44ad', width=2, dash='dashdot'), name='Düğüm Çizgisi'))
    fig.add_trace(go.Scatter3d(x=[Nx*3], y=[Ny*3], z=[0], mode='text', text=['N'], textfont=dict(size=14, color='#8e44ad'), textposition='top right', showlegend=False, hoverinfo='skip'))
    
    # Hover Metinleri (<extra></extra> ile trace etiketleri silindi)
    h_W = "<b>Çıkış Düğümü (Ω):</b><br>X ekseninden düğüm çizgisine.<extra></extra>"
    h_w = "<b>Enberi Argümanı (ω):</b><br>Düğümden Enberiye olan açı.<extra></extra>"
    h_i = "<b>Eğiklik (i):</b><br>Yörüngenin Ekvatora eğimi.<extra></extra>"
    h_nu = "<b>Gerçek Anomali (ν):</b><br>Enberiden anlık konuma.<extra></extra>"

    # YAYLAR VE HARFLER
    # Ω
    fig.add_trace(go.Scatter3d(x=X_W, y=Y_W, z=Z_W, mode='lines', line=dict(color=renk_W, width=4), name='Ω Yayı', legendgroup='W_yayi', hovertemplate=h_W))
    fig.add_trace(go.Scatter3d(x=[X_W[len(X_W)//2]], y=[Y_W[len(Y_W)//2]], z=[Z_W[len(Z_W)//2]], mode='markers+text', marker=dict(size=100, color='rgba(0,0,0,0)'), text=['Ω'], textfont=dict(size=18, color=renk_W), textposition='bottom center', showlegend=False, legendgroup='W_yayi', hovertemplate=h_W))
    
    # ω
    fig.add_trace(go.Scatter3d(x=X_w, y=Y_w, z=Z_w, mode='lines', line=dict(color=renk_w, width=4), name='ω Yayı', legendgroup='w_yayi', hovertemplate=h_w))
    fig.add_trace(go.Scatter3d(x=[X_w[len(X_w)//2]], y=[Y_w[len(Y_w)//2]], z=[Z_w[len(Z_w)//2]], mode='markers+text', marker=dict(size=100, color='rgba(0,0,0,0)'), text=['ω'], textfont=dict(size=18, color=renk_w), textposition='top center', showlegend=False, legendgroup='w_yayi', hovertemplate=h_w))
    
    # i
    fig.add_trace(go.Scatter3d(x=X_i, y=Y_i, z=Z_i, mode='lines', line=dict(color=renk_i, width=4), name='i Yayı', legendgroup='i_yayi', hovertemplate=h_i))
    fig.add_trace(go.Scatter3d(x=[X_i[len(X_i)//2]], y=[Y_i[len(Y_i)//2]], z=[Z_i[len(Z_i)//2]], mode='markers+text', marker=dict(size=100, color='rgba(0,0,0,0)'), text=['i'], textfont=dict(size=18, color=renk_i), textposition='middle right', showlegend=False, legendgroup='i_yayi', hovertemplate=h_i))

    # ν
    fig.add_trace(go.Scatter3d(x=X_nu, y=Y_nu, z=Z_nu, mode='lines', line=dict(color=renk_nu, width=4), name='ν Yayı', legendgroup='nu_yayi', hovertemplate=h_nu))
    fig.add_trace(go.Scatter3d(x=[X_nu[len(X_nu)//2]], y=[Y_nu[len(Y_nu)//2]], z=[Z_nu[len(Z_nu)//2]], mode='markers+text', marker=dict(size=100, color='rgba(0,0,0,0)'), text=['ν'], textfont=dict(size=18, color=renk_nu), textposition='bottom center', showlegend=False, legendgroup='nu_yayi', hovertemplate=h_nu))

    fig.update_layout(scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=40),
                      legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.1)", font=dict(size=10)))
    return fig
# ════════════════════════════════════════════════════════════════════
#  PDF OLUŞTURUCU MOTOR
# ════════════════════════════════════════════════════════════════════
def pdf_olustur(a, e, P, tau, cisim_ismi):
    t_g = np.linspace(tau, tau+P, 3000)
    _, _, _, r_g, x_g, y_g = hesapla(t_g, a, e, P, tau)
    r_enberi, r_enote = a*(1-e), a*(1+e)
    tablo_rows = anlamli_noktalar(a, e, P, tau)

    C_BASLIK, C_ALT, C_TH, C_ZEBRA = '#1a2940', '#2e6da4', '#1a2940', '#eaf2fb'
    FW, FH = 8.27, 11.69
    L, R, T, B = 0.055, 0.955, 0.968, 0.100

    pdf_buffer = io.BytesIO() # Fiziksel dosya yerine RAM'de oluştur (Web için şart)
    
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(FW, FH))
        METIN_BOT = 0.475
        ax = fig.add_axes([L, METIN_BOT, R-L, T-METIN_BOT])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

        AX_H, AX_W = (T - METIN_BOT) * FH, (R - L) * FW
        def pt2y(pt):   return (pt/72) / AX_H
        
        def yaz(x, y, s, fs=9, bold=False, renk='black', ha='left'):
            ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, 
                    fontweight='bold' if bold else 'normal', color=renk, ha=ha, va='top')
        def cizgi(y, renk='#cccccc', lw=0.7):
            ax.plot([0,1],[y,y], transform=ax.transAxes, color=renk, lw=lw)

        yp = 0.995
        yaz(0.5, yp, "GÖK MEKANİĞİ PROJE RAPORU", fs=14, bold=True, renk=C_BASLIK, ha='center')
        yp -= pt2y(14) + pt2y(5); cizgi(yp, renk=C_BASLIK, lw=1.0); yp -= pt2y(6)

        yaz(0.00, yp, "Ad Soyad: ", fs=9.5, renk='#222')
        yaz(0.44, yp, "Öğrenci No: ", fs=9.5, renk='#222')
        yaz(0.74, yp, "İmza:", fs=9.5, renk='#222')
        yp -= pt2y(13) + pt2y(7); cizgi(yp, renk='#dddddd'); yp -= pt2y(5)

        yaz(0.0, yp, "1.  RAPOR ÖZETİ", fs=9.5, bold=True, renk=C_ALT); yp -= pt2y(9.5) + pt2y(4)
        ozet = (f"{cisim_ismi} eliptik yörünge parametreleri (a = {a:.4f} AB, e = {e:.4f}, "
                f"P = {P:.2f} gün, τ = {jd_to_date(tau)}) girilerek Kepler denklemi sayısal olarak "
                f"çözülmüş; heliosentrik uzaklık (r) ve gerçek anomali (ν) hesaplanmıştır.")
        
        # Basit metin kaydırma
        kelimeler, satir, uzun, satirlar_o = ozet.split(), "", 0, []
        for k in kelimeler:
            if uzun + len(k) > 105:
                satirlar_o.append(satir.rstrip()); satir, uzun = k+" ", len(k)+1
            else:
                satir += k+" "; uzun += len(k)+1
        if satir.strip(): satirlar_o.append(satir.rstrip())
        for s in satirlar_o:
            yaz(0.02, yp, s, fs=8.6); yp -= pt2y(8.6) + pt2y(2.5)
        yp -= pt2y(3)

        yaz(0.0, yp, "2.  HESAPLAMA YÖNTEMİ", fs=9.5, bold=True, renk=C_ALT); yp -= pt2y(9.5) + pt2y(4)
        adimlar = [
            ("Ortalama anomali",          "M  =  (2π / P) · (t – τ)   mod 2π"),
            ("Başlangıç tahmini (N-R)",   "M < π → E₀ = M + e/2    |    M ≥ π → E₀ = M – e/2"),
            ("Newton-Raphson iter.",      "Eᵢ₊₁ = Eᵢ – f(Eᵢ)/f′(Eᵢ)   ,   durdurma: |f/f′| < 10⁻⁶"),
            ("Kepler fonksiyonları",      "f(E) = E – e·sinE – M      f′(E) = 1 – e·cosE"),
            ("Gerçek anomali (ν)",        "tan(ν/2) = √((1+e)/(1–e)) · tan(E/2)"),
            ("Heliosentrik uzaklık",      "r  =  a · (1 – e · cosE)"),
        ]
        for bas, form in adimlar:
            yaz(0.02, yp, f"▸  {bas}:", fs=8.6, bold=True, renk='#333')
            yaz(0.33, yp, form, fs=8.6, renk='#111'); yp -= pt2y(8.6) + pt2y(3)
        yp -= pt2y(4)

        yaz(0.0, yp, "3.  EFEMERİS TABLOSU  (Seçilmiş 6 Anlamlı Nokta)", fs=9.5, bold=True, renk=C_ALT)
        
        tablo_fig_bot = METIN_BOT + (yp - pt2y(9.5) - pt2y(7)) * (T - METIN_BOT) - 0.28 * (T - METIN_BOT)
        ax_t = fig.add_axes([L, tablo_fig_bot, R-L, 0.28 * (T - METIN_BOT)])
        ax_t.axis('off')

        rows_t = [[jd_to_date(t_n), etiket_isim, f"{M_n:.4f}", f"{E_n:.4f}", f"{np.degrees(nu_n):.2f}", f"{r_n:.4f}"] 
                  for (t_n, M_n, E_n, nu_n, r_n, etiket_isim) in tablo_rows]
        sutunlar = ["Tarih", "Nokta", "M (rad)", "E (rad)", "ν (°)", "r (AB)"]
        
        tbl = ax_t.table(cellText=rows_t, colLabels=sutunlar, loc='center', bbox=[0, 0, 1, 1])
        tbl.set_fontsize(7.8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#aaaaaa'); cell.set_linewidth(0.5)
            if row == 0: cell.set_facecolor(C_TH); cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0: cell.set_facecolor(C_ZEBRA)
            else: cell.set_facecolor('white')

        ax_p = fig.add_axes([L, B, R-L, 0.462-B])
        gezegen_ciz(ax_p, 0.387, 0.205, 'Merkür', '#aaaaaa')
        gezegen_ciz(ax_p, 0.723, 0.006, 'Venüs',  '#c8a200')
        gezegen_ciz(ax_p, 1.000, 0.016, 'Dünya',  '#2471a3')
        gezegen_ciz(ax_p, 1.524, 0.093, 'Mars',   '#c0392b')
        ax_p.plot(x_g, y_g, color='#1a2940', lw=1.6, label=f'{cisim_ismi} (e={e:.2f})', zorder=4)
        ax_p.scatter([0],[0], color='#f39c12', s=120, zorder=6, label='Güneş', edgecolors='#d68910', linewidths=0.7)

        idx_n, idx_t = np.argmin(r_g), np.argmax(r_g)
        ax_p.set_xlim(x_g.min()-0.55, x_g.max()+0.55); ax_p.set_ylim(y_g.min()-0.35, y_g.max()+0.35)

        def etiket(ax, xp, yp2, metin, renk, ox_pt, oy_pt):
            ax.annotate(metin, xy=(xp,yp2), xytext=(ox_pt, oy_pt), textcoords='offset points',
                        fontsize=8, color=renk, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=renk, lw=0.8, alpha=0.92),
                        arrowprops=dict(arrowstyle='->', color=renk, lw=0.8, shrinkA=0, shrinkB=3), zorder=7)

        etiket(ax_p, x_g[idx_n], y_g[idx_n], f"Enberi\n{r_enberi:.2f} AB", '#1e8449', 35, 35)
        etiket(ax_p, x_g[idx_t], y_g[idx_t], f"Enöte\n{r_enote:.2f} AB", '#7d3c98', -35, 35)
        ax_p.scatter([x_g[idx_n], x_g[idx_t]], [y_g[idx_n], y_g[idx_t]], color=['#1e8449', '#7d3c98'], s=45, zorder=5)

        ax_p.set_aspect('equal', 'box'); ax_p.set_xlabel('x  (AB)', fontsize=9); ax_p.set_ylabel('y  (AB)', fontsize=9)
        ax_p.set_title(f'Yörünge Diyagramı  —  a={a:.3f} AB,  e={e:.3f},  T={P:.2f} gün', fontsize=10, color=C_BASLIK, fontweight='bold', pad=5)
        ax_p.grid(True, linestyle=':', alpha=0.45, lw=0.55)
        ax_p.legend(loc='upper right', fontsize=7.5, framealpha=0.92, edgecolor='#cccccc')

        # --- QR KOD EKLEME ---
        site_url = "https://gokraporu.streamlit.app/" # Burayı Streamlit yayınlanınca güncelleyebilirsin
        qr = qrcode.QRCode(box_size=10, border=0)
        qr.add_data(site_url)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="black", back_color="white")
        
        ax_qr = fig.add_axes([0.85, 0.015, 0.09, 0.09]) 
        ax_qr.imshow(np.array(img_qr.convert('L')), cmap='gray')
        ax_qr.axis('off')
        ax_qr.text(0.5, 1.05, "Siz de kendi raporunuz oluşturmak\niçin sitemize gelebilirsiniz", 
                   transform=ax_qr.transAxes, ha='center', va='bottom', fontsize=6, color='#555')
        
        pdf.savefig(fig, dpi=220)
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer
def pdf_olustur_jenerik(a, e, i, W, w, nu, mu, R_vec, V_vec, cisim_ismi, merkez_ismi):
    C_BASLIK, C_ALT, C_TH, C_ZEBRA = '#1a2940', '#2e6da4', '#1a2940', '#eaf2fb'
    FW, FH = 8.27, 11.69
    L, R, T, B = 0.055, 0.955, 0.968, 0.100

    pdf_buffer = io.BytesIO()
    
    # Fiziksel hesaplamalar
    r_mag = np.linalg.norm(R_vec)
    v_mag = np.linalg.norm(V_vec)
    h_mag = np.linalg.norm(np.cross(R_vec, V_vec))
    epsilon = (v_mag**2 / 2) - (mu / r_mag)
    P = (2 * np.pi * np.sqrt(abs(a)**3 / mu)) if e < 1 else np.inf
    
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(FW, FH))
        
        # 1. BAŞLIK VE BİLGİLER
        ax_top = fig.add_axes([L, 0.8, R-L, 0.15])
        ax_top.axis('off')
        def yaz(x, y, s, fs=9, bold=False, renk='black', ha='left'):
            ax_top.text(x, y, s, transform=ax_top.transAxes, fontsize=fs, fontweight='bold' if bold else 'normal', color=renk, ha=ha, va='top')
        
        yaz(0.5, 0.95, "GÖK MEKANİĞİ LABORATUVAR RAPORU", fs=14, bold=True, renk=C_BASLIK, ha='center')
        ax_top.plot([0,1],[0.85,0.85], transform=ax_top.transAxes, color=C_BASLIK, lw=1.0)
        
        yaz(0.00, 0.77, "Ad Soyad: ", fs=9.5, renk='#222')
        yaz(0.44, 0.77, "Öğrenci No: ", fs=9.5, renk='#222')
        yaz(0.74, 0.77, "İmza:", fs=9.5, renk='#222')
        ax_top.plot([0,1],[0.68,0.68], transform=ax_top.transAxes, color='#dddddd', lw=0.7)

        yaz(0.0, 0.50, "1. HESAPLANAN PARAMETRELER", fs=10, bold=True, renk=C_ALT)
        
        # Tablo Çizimi (Vektörler ve Elemanlar)
        ax_t = fig.add_axes([L, 0.65, R-L, 0.12])
        ax_t.axis('off')
        
        tablo_veri = [
            ["Konum (X, Y, Z)", f"{R_vec[0]:.2f}, {R_vec[1]:.2f}, {R_vec[2]:.2f} km", "a (Yarı-büyük Eksen)", f"{a:.2f} km"],
            ["Hız (Vx, Vy, Vz)", f"{V_vec[0]:.3f}, {V_vec[1]:.3f}, {V_vec[2]:.3f} km/s", "e (Dışmerkezlik)", f"{e:.5f}"],
            ["Merkez Cisim (μ)", f"{mu:.2f} km³/s²", "i (Eğiklik)", f"{i:.2f}°"],
            ["", "", "Ω (Çıkış Düğümü)", f"{W:.2f}°"],
            ["", "", "ω (Enberi Argümanı)", f"{w:.2f}°"],
            ["", "", "ν (Gerçek Anomali)", f"{nu:.2f}°"]
        ]
        
        tbl = ax_t.table(cellText=tablo_veri, colLabels=["Durum Vektörleri", "Değer", "Yörünge Elemanları", "Değer"], loc='center', bbox=[0, 0, 1, 1])
        tbl.set_fontsize(8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#aaaaaa'); cell.set_linewidth(0.5)
            if row == 0: cell.set_facecolor(C_TH); cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0: cell.set_facecolor(C_ZEBRA)
            else: cell.set_facecolor('white')

        # 2. DİNAMİK FİZİKSEL BÜYÜKLÜKLER (ÖZET KUTUSU)
        ax_mid = fig.add_axes([L, 0.52, R-L, 0.1])
        ax_mid.axis('off')
        ax_mid.text(0.0, 0.9, "2. DİNAMİK BÜYÜKLÜKLER", transform=ax_mid.transAxes, fontsize=10, fontweight='bold', color=C_ALT, va='top')
        
        fizik_metin = (
            f"▸ Skaler Uzaklık (r) : {r_mag:.2f} km\n"
            f"▸ Skaler Hız (v) : {v_mag:.3f} km/s\n"
            f"▸ Özgül Açısal Momentum (h) : {h_mag:.2f} km²/s\n"
            f"▸ Özgül Mekanik Enerji (ε) : {epsilon:.4f} km²/s²\n"
            f"▸ Yörünge Periyodu (P) : {P/3600:.2f} saat ({P/86400:.2f} gün)" if e < 1 else f"▸ Yörünge Periyodu (P) : Açık Yörünge (Hiperbol/Parabol)"
        )
        ax_mid.text(0.02, 0.6, fizik_metin, transform=ax_mid.transAxes, fontsize=9, color='#222', va='top', linespacing=1.6)

        # 3. PERİFOKAL DÜZLEM ÇİZİMİ (2D)
        ax_p = fig.add_axes([L, 0.08, R-L, 0.40])
        ax_p.set_title("3. PERİFOKAL YÖRÜNGE DÜZLEMİ (x̄ - ȳ)", fontsize=10, color=C_BASLIK, fontweight='bold')
        
        # Yörünge elipsini çiz (True Anomaly üzerinden)
        nu_array = np.linspace(0, 2*np.pi, 500)
        p_param = a * (1 - e**2)
        r_array = p_param / (1 + e * np.cos(nu_array))
        x_bar = r_array * np.cos(nu_array)
        y_bar = r_array * np.sin(nu_array)
        
        ax_p.plot(x_bar, y_bar, color=C_BASLIK, lw=1.5, label='Yörünge', zorder=2)
        
        # Merkez cisim ve Enberi (Periapsis)
        merkez_renk = '#3498db' if merkez_ismi == "Dünya" else '#9b59b6'
        ax_p.scatter([0], [0], color=merkez_renk, s=120, zorder=5, label=merkez_ismi)
        ax_p.scatter([p_param/(1+e)], [0], color='#f39c12', s=50, zorder=5, label='Enberi (Π)')
        ax_p.plot([0, p_param/(1+e)], [0, 0], color='#f39c12', ls=':', lw=1.5, zorder=2)
        
        # Cismin anlık konumu (Perifokal)
        nu_rad = np.radians(nu)
        x_nu = r_mag * np.cos(nu_rad)
        y_nu = r_mag * np.sin(nu_rad)
        ax_p.scatter([x_nu], [y_nu], color='#e74c3c', s=60, zorder=5, label='Cisim (ν)')
        ax_p.plot([0, x_nu], [0, y_nu], color='#bdc3c7', ls='--', lw=1.5, zorder=1)
        
        # Hız vektörü oku (Perifokal Düzlemdeki İzdüşümü)
        vx_bar = -(mu / h_mag) * np.sin(nu_rad)
        vy_bar = (mu / h_mag) * (e + np.cos(nu_rad))
        scale = (a * 0.25) / v_mag # Oku yörünge boyutuna oranla
        ax_p.arrow(x_nu, y_nu, vx_bar*scale, vy_bar*scale, head_width=a*0.03, head_length=a*0.05, fc='#e74c3c', ec='#e74c3c', label='Hız Vektörü (v)', zorder=6)
        
        ax_p.set_aspect('equal', 'box')
        ax_p.grid(True, linestyle=':', alpha=0.5)
        ax_p.set_xlabel("x̄ (km)", fontsize=8)
        ax_p.set_ylabel("ȳ (km)", fontsize=8)
        
        # Legend
        handles, labels = ax_p.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_p.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, framealpha=0.9)

        pdf.savefig(fig, dpi=220)
        plt.close(fig)

def pdf_olustur_jenerik(a, e, i, W, w, nu, mu, R_vec, V_vec, cisim_ismi, merkez_ismi, mod="vektor"):
    C_BASLIK, C_ALT, C_KOYU = '#1a2940', '#2e6da4', '#111111'
    FW, FH = 8.27, 11.69
    
    pdf_buffer = io.BytesIO()
    
    r_mag = np.linalg.norm(R_vec)
    v_mag = np.linalg.norm(V_vec)
    vr = np.dot(R_vec, V_vec) / r_mag
    H_vec = np.cross(R_vec, V_vec)
    h_mag = np.linalg.norm(H_vec)
    K_vec = np.array([0, 0, 1])
    N_vec = np.cross(K_vec, H_vec)
    n_mag = np.linalg.norm(N_vec)
    E_vec = (1/mu) * ((v_mag**2 - mu/r_mag)*R_vec - r_mag*vr*V_vec)
    epsilon = (v_mag**2 / 2) - (mu / r_mag)
    P = (2 * np.pi * np.sqrt(abs(a)**3 / mu)) if e < 1 else np.inf

    with PdfPages(pdf_buffer) as pdf:
        # ==========================================
        # SAYFA 1: ADIM ADIM MATEMATİKSEL ÇÖZÜM
        # ==========================================
        fig1 = plt.figure(figsize=(FW, FH))
        ax1 = fig1.add_axes([0.08, 0.05, 0.84, 0.90])
        ax1.axis('off')
        
        def yaz(y, metin, fs=10, bold=False, renk=C_KOYU):
            ax1.text(0.0, y, metin, transform=ax1.transAxes, fontsize=fs, fontweight='bold' if bold else 'normal', color=renk, va='top', ha='left')
        
        ax1.text(0.5, 1.0, "GÖK MEKANİĞİ - ADIM ADIM ÇÖZÜM RAPORU", transform=ax1.transAxes, fontsize=14, fontweight='bold', color=C_BASLIK, ha='center', va='top')
        ax1.plot([0,1],[0.98,0.98], transform=ax1.transAxes, color=C_BASLIK, lw=1.5)
        
        yp = 0.95
        if mod == "vektor":
            yaz(yp, "PROBLEM: Verilen Durum Vektörlerinden Yörünge Elemanlarının Bulunması", fs=11, bold=True, renk=C_ALT); yp -= 0.03
            yaz(yp, "Verilenler:", bold=True); yp -= 0.02
            yaz(yp, r"μ = {mu} km³/s²"); yp -= 0.02
            yaz(yp, r"$\vec{r} = %.2f \hat{i} + %.2f \hat{j} + %.2f \hat{k} \quad (km)$" % (R_vec[0], R_vec[1], R_vec[2])); yp -= 0.02
            yaz(yp, r"$\vec{v} = %.4f \hat{i} + %.4f \hat{j} + %.4f \hat{k} \quad (km/s)$" % (V_vec[0], V_vec[1], V_vec[2])); yp -= 0.04
            
            yaz(yp, "ADIM 1: Skaler Büyüklükler ve Radyal Hız", bold=True); yp -= 0.02
            yaz(yp, r"$r = |\vec{r}| = \sqrt{x^2 + y^2 + z^2} = %.4f \ km$" % r_mag); yp -= 0.02
            yaz(yp, r"$v = |\vec{v}| = \sqrt{v_x^2 + v_y^2 + v_z^2} = %.4f \ km/s$" % v_mag); yp -= 0.02
            yaz(yp, r"$v_r = \frac{\vec{r} \cdot \vec{v}}{r} = %.4f \ km/s$" % vr); yp -= 0.04
            
            yaz(yp, "ADIM 2: Özgül Açısal Momentum Vektörü", bold=True); yp -= 0.02
            yaz(yp, r"$\vec{h} = \vec{r} \times \vec{v} = (y v_z - z v_y)\hat{i} + (z v_x - x v_z)\hat{j} + (x v_y - y v_x)\hat{k}$"); yp -= 0.02
            yaz(yp, r"$\vec{h} = %.2f \hat{i} + %.2f \hat{j} + %.2f \hat{k} \quad (km^2/s)$" % (H_vec[0], H_vec[1], H_vec[2])); yp -= 0.02
            yaz(yp, r"$h = |\vec{h}| = %.2f \ km^2/s$" % h_mag); yp -= 0.04
            
            yaz(yp, "ADIM 3: Eğiklik (i) ve Çıkış Düğümü (Ω)", bold=True); yp -= 0.02
            yaz(yp, r"$i = \arccos(h_z / h) = \arccos(%.2f / %.2f) = %.4f^\circ$" % (H_vec[2], h_mag, i)); yp -= 0.025
            yaz(yp, r"Düğüm Vektörü: $\vec{N} = \hat{k} \times \vec{h} = [-h_y, h_x, 0] = [%.2f, %.2f, 0]$" % (N_vec[0], N_vec[1])); yp -= 0.02
            yaz(yp, r"$n = |\vec{N}| = %.2f$" % n_mag); yp -= 0.025
            yaz(yp, r"$\Omega = \arccos(N_x / n) = %.4f^\circ$  (Eğer $N_y < 0$ ise $360 - \Omega$ alınır)" % W); yp -= 0.04
            
            yaz(yp, "ADIM 4: Dışmerkezlik Vektörü (e) ve Enberi Argümanı (ω)", bold=True); yp -= 0.02
            yaz(yp, r"$\vec{e} = \frac{1}{\mu} \left[ (v^2 - \frac{\mu}{r})\vec{r} - r v_r \vec{v} \right]$"); yp -= 0.02
            yaz(yp, r"$\vec{e} = [%.5f, %.5f, %.5f]$" % (E_vec[0], E_vec[1], E_vec[2])); yp -= 0.02
            yaz(yp, r"$e = |\vec{e}| = %.5f$" % e); yp -= 0.025
            yaz(yp, r"$\omega = \arccos \left( \frac{\vec{N} \cdot \vec{e}}{n e} \right) = %.4f^\circ$  (Eğer $e_z < 0$ ise $360 - \omega$)" % w); yp -= 0.04
            
            yaz(yp, "ADIM 5: Gerçek Anomali (ν) ve Yarı-Büyük Eksen (a)", bold=True); yp -= 0.02
            yaz(yp, r"$\nu = \arccos \left( \frac{\vec{e} \cdot \vec{r}}{e r} \right) = %.4f^\circ$  (Eğer $v_r < 0$ ise $360 - \nu$)" % nu); yp -= 0.025
            yaz(yp, r"$a = \frac{h^2}{\mu(1 - e^2)} = %.2f \ km$" % a); yp -= 0.04

        else:
            yaz(yp, "PROBLEM: Yörünge Elemanlarından Durum Vektörlerinin Bulunması", fs=11, bold=True, renk=C_ALT); yp -= 0.03
            yaz(yp, f"Verilenler: μ={mu}, a={a:.2f}, e={e:.4f}, i={i:.2f}°, Ω={W:.2f}°, ω={w:.2f}°, ν={nu:.2f}°"); yp -= 0.04
            
            yaz(yp, "ADIM 1: Perifokal Düzlemdeki Uzaklık ve Koordinatlar", bold=True); yp -= 0.02
            p_val = a * (1 - e**2)
            yaz(yp, r"$p = a(1 - e^2) = %.2f \ km$" % p_val); yp -= 0.02
            yaz(yp, r"$r = \frac{p}{1 + e \cos\nu} = %.2f \ km$" % r_mag); yp -= 0.03
            yaz(yp, r"$\bar{x} = r \cos\nu = %.2f \ km$" % (r_mag * np.cos(np.radians(nu)))); yp -= 0.02
            yaz(yp, r"$\bar{y} = r \sin\nu = %.2f \ km$" % (r_mag * np.sin(np.radians(nu)))); yp -= 0.04

            yaz(yp, "ADIM 2: Perifokal Düzlemdeki Hız Bileşenleri", bold=True); yp -= 0.02
            h_val = np.sqrt(mu * p_val)
            yaz(yp, r"$h = \sqrt{\mu p} = %.2f \ km^2/s$" % h_val); yp -= 0.02
            yaz(yp, r"$\bar{v}_x = -\frac{\mu}{h} \sin\nu = %.4f \ km/s$" % (-(mu/h_val)*np.sin(np.radians(nu)))); yp -= 0.02
            yaz(yp, r"$\bar{v}_y = \frac{\mu}{h} (e + \cos\nu) = %.4f \ km/s$" % ((mu/h_val)*(e + np.cos(np.radians(nu))))); yp -= 0.04

            yaz(yp, "ADIM 3: 3B Ekvatoryal Uzaya Dönüşüm Matrisi (Euler Açıları)", bold=True); yp -= 0.02
            yaz(yp, r"$R_{313}(\Omega, i, \omega) = R_3(-\Omega) R_1(-i) R_3(-\omega)$"); yp -= 0.03
            yaz(yp, r"Dönüşüm sonrasında Ekvatoryal Uzaydaki (X, Y, Z) Durum Vektörleri:"); yp -= 0.02
            yaz(yp, r"$\vec{R} = [%.2f, \ %.2f, \ %.2f] \ km$" % (R_vec[0], R_vec[1], R_vec[2])); yp -= 0.02
            yaz(yp, r"$\vec{V} = [%.4f, \ %.4f, \ %.4f] \ km/s$" % (V_vec[0], V_vec[1], V_vec[2])); yp -= 0.04
            
        pdf.savefig(fig1, dpi=220)
        plt.close(fig1)

        # ==========================================
        # SAYFA 2: GRAFİK VE FİZİKSEL ÖZET
        # ==========================================
        fig2 = plt.figure(figsize=(FW, FH))
        
        # Fiziksel Özet Tablosu
        ax_top = fig2.add_axes([0.1, 0.75, 0.8, 0.15])
        ax_top.axis('off')
        ax_top.text(0.0, 1.0, "EK: SİSTEMİN FİZİKSEL DURUM ÖZETİ", transform=ax_top.transAxes, fontsize=11, fontweight='bold', color=C_ALT)
        fizik_metin = (
            f"▸ Skaler Uzaklık (r) : {r_mag:.2f} km\n"
            f"▸ Skaler Hız (v) : {v_mag:.4f} km/s\n"
            f"▸ Özgül Açısal Momentum (h) : {h_mag:.2f} km²/s\n"
            f"▸ Özgül Mekanik Enerji (ε) : {epsilon:.4f} km²/s²\n"
            f"▸ Yörünge Periyodu (P) : {P/3600:.2f} saat ({P/86400:.2f} gün)" if e < 1 else f"▸ Yörünge Periyodu (P) : Açık Yörünge (Hiperbol/Parabol)"
        )
        ax_top.text(0.02, 0.7, fizik_metin, transform=ax_top.transAxes, fontsize=10, linespacing=1.8, va='top')

        # Perifokal Grafik
        ax_p = fig2.add_axes([0.15, 0.15, 0.7, 0.5])
        ax_p.set_title("Perifokal Yörünge Düzlemi İzdüşümü (x̄ - ȳ)", fontsize=10, color=C_BASLIK, fontweight='bold', pad=15)
        
        nu_array = np.linspace(0, 2*np.pi, 500)
        p_param = a * (1 - e**2)
        r_array = p_param / (1 + e * np.cos(nu_array))
        x_bar = r_array * np.cos(nu_array)
        y_bar = r_array * np.sin(nu_array)
        
        ax_p.plot(x_bar, y_bar, color=C_BASLIK, lw=1.5, label='Yörünge', zorder=2)
        
        merkez_renk = '#3498db' if merkez_ismi == "Dünya" else '#9b59b6'
        ax_p.scatter([0], [0], color=merkez_renk, s=120, zorder=5, label=merkez_ismi)
        ax_p.scatter([p_param/(1+e)], [0], color='#f39c12', s=50, zorder=5, label='Enberi (Π)')
        ax_p.plot([0, p_param/(1+e)], [0, 0], color='#f39c12', ls=':', lw=1.5, zorder=2)
        
        nu_rad = np.radians(nu)
        x_nu = r_mag * np.cos(nu_rad)
        y_nu = r_mag * np.sin(nu_rad)
        ax_p.scatter([x_nu], [y_nu], color='#e74c3c', s=60, zorder=5, label='Cisim (ν)')
        ax_p.plot([0, x_nu], [0, y_nu], color='#bdc3c7', ls='--', lw=1.5, zorder=1)
        
        vx_bar = -(mu / h_mag) * np.sin(nu_rad)
        vy_bar = (mu / h_mag) * (e + np.cos(nu_rad))
        scale = (a * 0.25) / v_mag 
        ax_p.arrow(x_nu, y_nu, vx_bar*scale, vy_bar*scale, head_width=a*0.04, head_length=a*0.06, fc='#e74c3c', ec='#e74c3c', label='Hız Vektörü (v)', zorder=6)
        
        ax_p.set_aspect('equal', 'box')
        ax_p.grid(True, linestyle=':', alpha=0.5)
        ax_p.set_xlabel("x̄ (km)", fontsize=9)
        ax_p.set_ylabel("ȳ (km)", fontsize=9)
        handles, labels = ax_p.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_p.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, framealpha=0.9)

        pdf.savefig(fig2, dpi=220)
        plt.close(fig2)

    pdf_buffer.seek(0)
    return pdf_buffer
    
# ════════════════════════════════════════════════════════════════════
#  ANA ARAYÜZ (UI)
# ════════════════════════════════════════════════════════════════════
if "secim" not in st.session_state:
    st.session_state.secim = None

st.markdown("<div style='text-align: center; font-size: 2.4em; font-weight: 700;'>Otonom Gök Mekaniği Raporlayıcısı 🪐</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.1em; opacity: 0.85; margin-top: 10px;'>Güneş sistemi cisimlerinin yörünge dinamiklerini hesaplayın, efemeris tablolarını oluşturun ve tek tıkla akademik formatta PDF raporları elde edin.</div><hr>", unsafe_allow_html=True)

# Buton rengini #008b8b yapmak için CSS müdahalesi
if "merkez_tipi" not in st.session_state:
    st.session_state.merkez_tipi = None

# Buton rengini #1293B4 yapmak için CSS müdahalesi
st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1293B4;
        color: white;
        border-color: #1293B4;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0f7a96;
        border-color: #0f7a96;
    }
    </style>
    """, unsafe_allow_html=True)

if st.session_state.secim is None:
    st.markdown("<br><div style='text-align: center; font-size: 1.4em; font-weight: 600;'>Referans Sisteminizi ve Merkez Cismi Seçin</div><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("☀️\n\nGÜNEŞ MERKEZLİ\n\n(Heliosentrik - Ekliptik)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "gunes_secim"
            st.session_state.merkez_tipi = "gunes"
            st.rerun()
    with col2:
        if st.button("🌍\n\nDÜNYA MERKEZLİ\n\n(Jeosentrik - Ekvatoryal)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "jenerik_manuel"
            st.session_state.merkez_tipi = "dunya"
            st.rerun()
    with col3:
        if st.button("🪐\n\nÖZEL CİSİM MERKEZLİ\n\n(Serbest Laboratuvar)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "jenerik_manuel"
            st.session_state.merkez_tipi = "ozel"
            st.rerun()

# --- GÜNEŞ MERKEZLİ ALT MENÜ ---
if st.session_state.secim == "gunes_secim":
    if st.button("← Ana Menüye Dön"):
        st.session_state.secim = None
        st.session_state.merkez_tipi = None
        st.rerun()
    st.markdown("<div style='text-align: center; font-size: 1.2em; margin-bottom: 20px;'>Veri giriş yöntemini seçin:</div>", unsafe_allow_html=True)
    alt_col1, alt_col2 = st.columns(2)
    with alt_col1:
        if st.button("📡\n\nJPL VERİTABANI\n\n(Otomatik Çek)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "jpl"
            st.rerun()
    with alt_col2:
        if st.button("✍️\n\nMANUEL GİRİŞ\n\n(Temel Parametreler)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "manuel"
            st.rerun()
# --- DÜNYA VE ÖZEL CİSİM EKRANI (ÇİFT YÖNLÜ MOTOR) ---
elif st.session_state.secim == "jenerik_manuel":
    if st.button("← Ana Menüye Dön"):
        st.session_state.secim = None
        st.session_state.merkez_tipi = None
        for key in ['aktif_fig', 'aktif_isim']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
        
    st.info("💡 Çift Yönlü Motora Hoş Geldiniz! İster konum/hız vektörlerini, ister açıları girin; motor eksik olanı tamamlar.")
    
    mu_val = 398600.4418
    merkez_isim_etiket = "Dünya"
    
    if st.session_state.merkez_tipi == "ozel":
        merkez_isim_etiket = "Özel Merkez"
        mu_val = st.number_input("Merkez Cismin Standart Kütleçekim Parametresi (μ) [km³/s²]:", value=398600.4418, format="%.4f")
    else:
        st.markdown(f"**Aktif Merkez:** Dünya (μ = {mu_val} km³/s²)")

    tab_vektor, tab_eleman = st.tabs(["🚀 Durum Vektörleri (r, v) Gireceğim", "📐 Yörünge Elemanları Gireceğim"])
    
    with tab_vektor:
        st.markdown("Konum ve hız bileşenlerini girerek Kepler açılarını hesaplayın.")
        col_r1, col_r2, col_r3 = st.columns(3)
        rx = col_r1.number_input("X Konumu [km]", value=-6045.0)
        ry = col_r2.number_input("Y Konumu [km]", value=-3490.0)
        rz = col_r3.number_input("Z Konumu [km]", value=2500.0)
        
        col_v1, col_v2, col_v3 = st.columns(3)
        vx = col_v1.number_input("Vx Hızı [km/s]", value=-3.457)
        vy = col_v2.number_input("Vy Hızı [km/s]", value=6.618)
        vz = col_v3.number_input("Vz Hızı [km/s]", value=2.533)
        
        if st.button("Geometriyi Çöz ve Çiz (Vektör Modu)", type="primary"):
            with st.spinner("Vektörler çözümleniyor ve Akademik PDF Raporu hazırlanıyor..."):
                R = np.array([rx, ry, rz]); V = np.array([vx, vy, vz])
                a_c, e_c, i_c, W_c, w_c, nu_c = vektor_to_eleman(R, V, mu_val)
                st.success(f"Hesaplanan Elemanlar: a={a_c:.1f}km, e={e_c:.4f}, i={i_c:.2f}°, Ω={W_c:.2f}°, ω={w_c:.2f}°, ν={nu_c:.2f}°")
                st.session_state.aktif_fig = plotly_3d_ciz_jenerik(a_c, e_c, i_c, W_c, w_c, nu_c, mu_val, "Uydu/Cisim", merkez_isim_etiket)
                st.session_state.aktif_pdf_jenerik = pdf_olustur_jenerik(a_c, e_c, i_c, W_c, w_c, nu_c, mu_val, R, V, "Uydu/Cisim", merkez_isim_etiket, mod="vektor")

    with tab_eleman:
        st.markdown("**(PDF Sayfa 7-8 Formatı)** - Kepler elemanlarını girerek durum vektörlerini hesaplayın.")
        col_e1, col_e2, col_e3 = st.columns(3)
        a_in = col_e1.number_input("Yarı-Büyük Eksen (a) [km]", value=8788.0)
        e_in = col_e2.number_input("Dışmerkezlik (e)", value=0.1712, format="%.4f")
        nu_in= col_e3.number_input("Gerçek Anomali (ν) [°]", value=28.45)
        
        col_e4, col_e5, col_e6 = st.columns(3)
        i_in = col_e4.number_input("Eğiklik (i) [°]", value=153.2)
        W_in = col_e5.number_input("Çıkış Düğümü (Ω) [°]", value=255.3)
        w_in = col_e6.number_input("Enberi Arg. (ω) [°]", value=20.07)
        
        if st.button("Vektörleri Bul ve Çiz (Eleman Modu)", type="primary"):
            with st.spinner("Dönüşüm matrisleri hesaplanıyor ve Akademik PDF Raporu hazırlanıyor..."):
                R_out, V_out, r_mag = eleman_to_vektor(a_in, e_in, i_in, W_in, w_in, nu_in, mu_val)
                st.success(f"Hesaplanan Vektörler:\nR: [{R_out[0]:.1f}, {R_out[1]:.1f}, {R_out[2]:.1f}] km\nV: [{V_out[0]:.3f}, {V_out[1]:.3f}, {V_out[2]:.3f}] km/s")
                st.session_state.aktif_fig = plotly_3d_ciz_jenerik(a_in, e_in, i_in, W_in, w_in, nu_in, mu_val, "Uydu/Cisim", merkez_isim_etiket)
                st.session_state.aktif_pdf_jenerik = pdf_olustur_jenerik(a_in, e_in, i_in, W_in, w_in, nu_in, mu_val, R_out, V_out, "Uydu/Cisim", merkez_isim_etiket, mod="eleman")

    if "aktif_fig" in st.session_state and st.session_state.secim == "jenerik_manuel":
        st.plotly_chart(st.session_state.aktif_fig, use_container_width=True)
        
        # EĞER PDF OLUŞTUYSA İNDİR BUTONUNU GÖSTER
        if "aktif_pdf_jenerik" in st.session_state:
            st.download_button(
                label="📥 PDF Laboratuvar Raporunu İndir",
                data=st.session_state.aktif_pdf_jenerik,
                file_name="GokMekanigi_Vektor_Raporu.pdf",
                mime="application/pdf"
            )
elif st.session_state.secim == "jpl":
    if st.button("← Geri Dön / Yöntem Değiştir"):
        st.session_state.secim = None
        for key in ['aktif_pdf', 'aktif_fig', 'aktif_isim']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
        
    st.info("💡 JPL Horizons veritabanı kullanılarak parametreler otomatik çekilir. "
            "Emin olmadığınız cisim isimlerini [NASA JPL Horizons](https://ssd.jpl.nasa.gov/horizons/app.html#/) "
            "üzerinden kontrol edebilirsiniz. Örn: '2010 LP33', 'Ceres', 'Halley'")
    cisim_adi = st.text_input("Gök Cisminin Adı:", placeholder="Örn: 2010 LP33")
    
    if st.button("Raporu Oluştur 📝"):
        if not cisim_adi:
            st.warning("Lütfen bir gök cismi adı girin.")
        else:
            with st.spinner("JPL Horizons veritabanına bağlanılıyor..."):
                try:
                    obj = Horizons(id=cisim_adi, location='@sun', epochs=None)
                    el = obj.elements()
                    if el is not None and len(el) > 0:
                        a_val = float(el['a'][0])
                        e_val = float(el['e'][0])
                        tau_val = float(el['Tp_jd'][0])
                        try:
                            P_val = float(el['P'][0])
                        except:
                            P_val = (a_val**1.5) * 365.256
                        i_val = float(el['incl'][0])
                        W_val = float(el['Omega'][0])
                        w_val = float(el['w'][0])

                        st.session_state.aktif_pdf = pdf_olustur(a_val, e_val, P_val, tau_val, cisim_ismi=f"Asteroit {cisim_adi.upper()}")
                        st.session_state.aktif_fig = plotly_3d_ciz(a_val, e_val, P_val, tau_val, i_val, W_val, w_val, cisim_adi.upper())
                        st.session_state.aktif_isim = cisim_adi.upper()
                    else:
                        st.error("Gök cismi bulunamadı. Lütfen ismi kontrol edip tekrar deneyin.")
                except Exception as e:
                    st.error(f"Sunucularında hata oluştu veya cisim bulunamadı. Lütfen manuel girişi deneyin. Hata detayı: {str(e)}")

    if "aktif_pdf" in st.session_state and st.session_state.secim == "jpl":
        st.success(f"Raporunuz başarıyla hazırlandı! ({st.session_state.aktif_isim})")
        st.plotly_chart(st.session_state.aktif_fig, use_container_width=True)
        st.download_button(
            label="📥 PDF Raporunu İndir",
            data=st.session_state.aktif_pdf,
            file_name=f"{st.session_state.aktif_isim}_Raporu.pdf",
            mime="application/pdf"
        )

elif st.session_state.secim == "manuel":
    if st.button("← Geri Dön / Yöntem Değiştir"):
        st.session_state.secim = None
        for key in ['aktif_pdf', 'aktif_fig', 'aktif_isim']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.info("💡 Lütfen temel yörünge parametrelerini eksiksiz girin. 3D simülasyon için uzaysal parametreleri de ekleyebilirsiniz.")
    
    st.markdown("#### Temel Parametreler (Zorunlu)")
    col1, col2 = st.columns(2)
    with col1:
        a_val = st.number_input("Yarı-büyük eksen (a) [AB]", min_value=0.001, value=None, placeholder="Örn: 2.55", step=0.1)
        e_val = st.number_input("Dışmerkezlik (e)", min_value=0.0, max_value=0.999, value=None, placeholder="Örn: 0.42", step=0.01)
    with col2:
        P_val = st.number_input("Dönem (P) [Gün]", min_value=0.1, value=None, placeholder="Örn: 1491.04", step=10.0)
        tau_val = st.number_input("Enberiden geçiş (τ) [JD]", value=None, placeholder="Örn: 2458344.234332", step=100.0, format="%.6f")
        
    st.markdown("#### Uzaysal Parametreler (3D Simülasyon İçin İsteğe Bağlı)")
    col3, col4, col5 = st.columns(3)
    with col3:
        i_val = st.number_input("Eğiklik (i) [°]", min_value=0.0, max_value=180.0, value=None, placeholder="Örn: 10.5", step=1.0)
    with col4:
        W_val = st.number_input("Çıkış Düğümü (Ω) [°]", min_value=0.0, max_value=360.0, value=None, placeholder="Örn: 80.3", step=1.0)
    with col5:
        w_val = st.number_input("Enberi Arg. (ω) [°]", min_value=0.0, max_value=360.0, value=None, placeholder="Örn: 73.1", step=1.0)
        
    if st.button("Raporu Oluştur 📝"):
        if None in [a_val, e_val, P_val, tau_val]:
            st.warning("Lütfen raporu oluşturmadan önce Temel Parametreleri eksiksiz doldurun.")
        else:
            uzaysal_girdiler = [i_val, W_val, w_val]
            uzaysal_dolu_sayisi = sum(x is not None for x in uzaysal_girdiler)
            
            if uzaysal_dolu_sayisi > 0 and uzaysal_dolu_sayisi < 3:
                st.warning("⚠️ 3D simülasyon oluşturabilmek için i, Ω ve ω değerlerinin üçü de girilmelidir veya üçü de boş bırakılmalıdır.")
            else:
                with st.spinner("Hesaplanıyor ve Çiziliyor..."):
                    st.session_state.aktif_pdf = pdf_olustur(a=a_val, e=e_val, P=P_val, tau=tau_val, cisim_ismi="Özel Gök Cismi")
                    st.session_state.aktif_isim = "Özel_Gok_Cismi"
                    
                    if uzaysal_dolu_sayisi == 3:
                        st.session_state.aktif_fig = plotly_3d_ciz(a_val, e_val, P_val, tau_val, i_val, W_val, w_val, "Özel Gök Cismi")
                    elif "aktif_fig" in st.session_state:
                        del st.session_state["aktif_fig"]

    if "aktif_pdf" in st.session_state and st.session_state.secim == "manuel":
        st.success("Raporunuz başarıyla hazırlandı!")
        
        if "aktif_fig" in st.session_state:
            st.plotly_chart(st.session_state.aktif_fig, use_container_width=True)
            
        st.download_button(
            label="📥 PDF Raporunu İndir",
            data=st.session_state.aktif_pdf,
            file_name=f"{st.session_state.aktif_isim}_Raporu.pdf",
            mime="application/pdf"
        )
# ════════════════════════════════════════════════════════════════════
#  FOOTER (GELİŞTİRİCİ VİZYONU)
# ════════════════════════════════════════════════════════════════════
st.markdown("---")

with st.expander("Geliştirici Vizyonu 🚀", expanded=True):
    st.markdown(
        "Ankara Üniversitesi Astronomi ve Uzay Bilimleri bölümünde öğrenim görmekte ve teorik astronominin "
        "pek çok noktasıyla ilgilenmekteyim. Python kullanarak geliştirdiğim "
        "matematiksel modellemeleri ve bilimsel veri analizi araçlarını, herkesin erişebileceği dinamik "
        "uygulamalara dönüştürmeyi hedefliyorum.\n\n"
        "Başlangıçta yörünge parametrelerinin sayısal analizi ve "
        "Kepler denkleminin çözümü için kurguladığım bu Python tabanlı gök mekaniği motorunu, arayüz "
        "tasarımında vakit kaybetmemek ve odağı tamamen işlevsellikte tutmak adına modern yapay zeka "
        "araçları yardımıyla otonom bir efemeris raporlayıcısına çevirdim.\n\n"
        "Amacım, karmaşık bilimsel hesaplamaları hantal süreçlerden kurtarıp hızlı, otonom ve kullanıcı "
        "dostu araçlar haline getirmektir."
    )
