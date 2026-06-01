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

# ════════════════════════════════════════════════════════════════════
#  ANA ARAYÜZ (UI)
# ════════════════════════════════════════════════════════════════════
if "secim" not in st.session_state:
    st.session_state.secim = None

st.markdown("<div style='text-align: center; font-size: 2.4em; font-weight: 700;'>Otonom Gök Mekaniği Raporlayıcısı 🪐</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.1em; opacity: 0.85; margin-top: 10px;'>Güneş sistemi cisimlerinin yörünge dinamiklerini hesaplayın, efemeris tablolarını oluşturun ve tek tıkla akademik formatta PDF raporları elde edin.</div><hr>", unsafe_allow_html=True)

# Buton rengini #008b8b yapmak için CSS müdahalesi
st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #008b8b;
        color: white;
        border-color: #008b8b;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #006b6b;
        border-color: #006b6b;
    }
    </style>
    """, unsafe_allow_html=True)

if st.session_state.secim is None:
    st.markdown("<br><div style='text-align: center; font-size: 1.4em; font-weight: 600;'>Hangi yöntemle rapor oluşturmak istersiniz?</div><br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🛰️\n\nGÖK CİSMİNİN İSMİYLE\n\n(Otomatik JPL Bağlantısı)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "jpl"
            st.rerun()
    with col2:
        if st.button("🧮\n\nPARAMETRELERLE\n\n(Manuel Giriş)\n\n", use_container_width=True, type="primary"):
            st.session_state.secim = "manuel"
            st.rerun()

if st.session_state.secim == "jpl":
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
