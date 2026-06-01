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

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
#  SAYFA AYARLARI VE YAN MENÜ (SIDEBAR)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Gök Mekaniği Raporlayıcı", layout="centered", page_icon="🪐")

st.sidebar.title("Hakkımda")
st.sidebar.info(
    "**[Buraya Kendi Adını Yaz]**\n\n"
    "[Kendini tanıttığın, astronomi ve uzay bilimleri üzerine vizyonunu "
    "veya bu projeyi yapma amacını anlatan kısa bir metin gir.]"
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Tüm Hakları Saklıdır. [Proje/Ders Adı İçin Geliştirilmiştir]")

# ════════════════════════════════════════════════════════════════════
#  HESAPLAMA FONKSİYONLARI (Önceki kodun aynısı, Streamlit uyumlu)
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
    L, R, T, B = 0.055, 0.955, 0.968, 0.080

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

        imza_rect = plt.Rectangle((L + 0.80*(R-L), METIN_BOT + (yp - pt2y(13)) * (T - METIN_BOT)), 
                                  R - 0.008 - (L + 0.80*(R-L)), pt2y(13) * (T - METIN_BOT),
                                  linewidth=0.9, edgecolor='#666', facecolor='#fafafa',
                                  transform=fig.transFigure, zorder=5, clip_on=False)
        fig.add_artist(imza_rect)

        yp -= pt2y(13) + pt2y(7); cizgi(yp, renk='#dddddd'); yp -= pt2y(5)

        yaz(0.0, yp, "1.  PROJE ÖZETİ", fs=9.5, bold=True, renk=C_ALT); yp -= pt2y(9.5) + pt2y(4)
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
        
        ax_qr = fig.add_axes([0.80, 0.02, 0.10, 0.02]) # Sağ alt köşeye hizala
        ax_qr.imshow(np.array(img_qr.convert('L')), cmap='gray')
        ax_qr.axis('off')
        ax_qr.text(0.5, 1.05, "Siz de kendi raporunuz oluşturmak\niçin sitemize gelebilirsiniz", 
                   transform=ax_qr.transAxes, ha='center', va='bottom', fontsize=6, color='#555')
        
        pdf.savefig(fig, dpi=220)
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer

# ════════════════════════════════════════════════════════════════════
#  WEB ARAYÜZÜ (STREAMLIT)
# ════════════════════════════════════════════════════════════════════
st.title("Hoş Geldiniz! 🔭")
st.markdown("Hangi şekilde efemeris tablosu ve yörünge hareketini çizdirmek isterdiniz?")

secim = st.radio("Lütfen bir yöntem seçin:", 
                 ["Cisim İsmi İle (Otomatik JPL Bağlantısı)", 
                  "Yörünge Parametrelerini Kendim Gireceğim"], 
                 index=None)

if secim == "Cisim İsmi İle (Otomatik JPL Bağlantısı)":
    st.info("💡 JPL Horizons veritabanı kullanılarak parametreler otomatik çekilir. Örneğin: '2010 LP33', 'Ceres', 'Halley'")
    cisim_adi = st.text_input("Gök Cisminin Adı:", placeholder="Örn: 2010 LP33")
    
    if st.button("Raporu Oluştur 📝"):
        if not cisim_adi:
            st.warning("Lütfen bir cisim adı girin.")
        else:
            with st.spinner("JPL Horizons veritabanına bağlanılıyor..."):
                try:
                    obj = Horizons(id=cisim_adi, location='@sun')
                    el = obj.elements()
                    
                    if len(el) == 0:
                        st.error("Cisim bulunamadı. Lütfen ismi kontrol edip tekrar deneyin.")
                    else:
                        # JPL'den verileri ayıkla
                        a = float(el['a'][0])
                        e = float(el['e'][0])
                        tau = float(el['Tp_jd'][0])
                        P = (a**1.5) * 365.256 # JPL bazen P'yi vermezse Kepler yasasıyla hesaplanır
                        
                        pdf_data = pdf_olustur(a, e, P, tau, cisim_ismi=f"Asteroit {cisim_adi.upper()}")
                        st.success(f"Raporunuz başarıyla hazırlandı! ({cisim_adi})")
                        
                        st.download_button(
                            label="📥 PDF Raporunu İndir",
                            data=pdf_data,
                            file_name=f"{cisim_adi.replace(' ', '_')}_Raporu.pdf",
                            mime="application/pdf"
                        )
                except Exception as ex:
                    st.error(f"Sunucularında hata oluştu veya cisim bulunamadı. Lütfen manuel girişi deneyin. Hata detayı: {ex}")

elif secim == "Yörünge Parametrelerini Kendim Gireceğim":
    st.info("💡 Lütfen yörünge parametrelerini eksiksiz girin.")
    col1, col2 = st.columns(2)
    with col1:
        a_val = st.number_input("Yarı-büyük eksen (a) [AB]", min_value=0.001, value=None, placeholder="Örn: 2.55", step=0.1)
        e_val = st.number_input("Dışmerkezlik (e)", min_value=0.0, max_value=0.999, value=None, placeholder="Örn: 0.42", step=0.01)
    with col2:
        P_val = st.number_input("Dönem (P) [Gün]", min_value=0.1, value=None, placeholder="Örn: 1491.04", step=10.0)
        tau_val = st.number_input("Enberiden geçiş (τ) [JD]", value=None, placeholder="Örn: 2458344.234332", step=100.0, format="%.6f")
        
    if st.button("Raporu Oluştur 📝"):
        if None in [a_val, e_val, P_val, tau_val]:
            st.warning("Lütfen raporu oluşturmadan önce tüm parametreleri doldurun.")
        else:
            with st.spinner("Hesaplanıyor ve Çiziliyor..."):
                pdf_data = pdf_olustur(a=a_val, e=e_val, P=P_val, tau=tau_val, cisim_ismi="Özel Gök Cismi")
            st.success("Raporunuz başarıyla hazırlandı!")
            
            st.download_button(
                label="📥 PDF Raporunu İndir",
                data=pdf_data,
                file_name="Gok_Mekanigi_Ozel_Rapor.pdf",
                mime="application/pdf"
            )
