import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HydroSense — Chennai Reservoir Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* Custom cursor */
html { cursor: none !important; }
.cursor-dot {
    width: 6px; height: 6px;
    background: #00C8FF;
    border-radius: 50%;
    position: fixed;
    pointer-events: none;
    z-index: 99999;
    transition: transform 0.1s ease;
    transform: translate(-50%, -50%);
}
.cursor-ring {
    width: 32px; height: 32px;
    border: 1.5px solid rgba(0,200,255,0.5);
    border-radius: 50%;
    position: fixed;
    pointer-events: none;
    z-index: 99998;
    transition: transform 0.15s ease, width 0.2s, height 0.2s, border-color 0.2s;
    transform: translate(-50%, -50%);
}

/* Root */
html, body, .stApp {
    background: #040A12 !important;
    font-family: 'Space Grotesk', sans-serif;
    color: #D6E8F5;
    min-height: 100vh;
}

/* Animated water wave background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 40% at 20% 80%, rgba(0,80,180,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 30% at 80% 20%, rgba(0,200,255,0.06) 0%, transparent 60%),
        linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 50px 50px, 50px 50px;
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; position: relative; z-index: 1; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00C8FF;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.2rem;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: '';
    width: 28px; height: 1px;
    background: rgba(0,200,255,0.4);
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5.5vw, 4rem);
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 0%, #80E8FF 45%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.9rem;
}
.hero p {
    color: rgba(214,232,245,0.5);
    font-size: 1rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
    font-weight: 400;
}
.hero-divider {
    width: 1px;
    height: 40px;
    background: linear-gradient(180deg, transparent, #00C8FF, transparent);
    margin: 1.8rem auto 0;
}

/* ── Stats Row ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-item {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 16px;
    padding: 1.1rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-item::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 60%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.4), transparent);
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    background: linear-gradient(135deg, #80E8FF, #0099FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-lbl {
    font-size: 0.68rem;
    color: rgba(214,232,245,0.38);
    margin-top: 3px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), transparent);
}
.card:hover {
    border-color: rgba(0,200,255,0.2);
    box-shadow: 0 8px 40px rgba(0,100,200,0.08);
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00C8FF;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,200,255,0.15);
}

/* ── Inputs ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00C8FF, #0055CC) !important;
}
.stSlider > div > div > div > div > div {
    background: #00C8FF !important;
    border: 3px solid #040A12 !important;
    box-shadow: 0 0 14px rgba(0,200,255,0.7) !important;
    width: 20px !important; height: 20px !important;
}
.stSlider [data-baseweb="slider"] > div:first-child {
    background: rgba(255,255,255,0.07) !important;
}
.stSlider p { color: rgba(214,232,245,0.65) !important; font-size: 0.88rem !important; }

.stSelectbox > label { color: rgba(214,232,245,0.65) !important; font-size: 0.88rem !important; }
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 12px !important;
    color: #D6E8F5 !important;
}
.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(0,200,255,0.4) !important;
}

.stNumberInput > label { color: rgba(214,232,245,0.65) !important; font-size: 0.88rem !important; }
.stNumberInput input {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 12px !important;
    color: #D6E8F5 !important;
    font-size: 0.95rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stNumberInput input:focus {
    border-color: rgba(0,200,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,200,255,0.08) !important;
}

.stDateInput > label { color: rgba(214,232,245,0.65) !important; font-size: 0.88rem !important; }
.stDateInput input {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 12px !important;
    color: #D6E8F5 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00C8FF 0%, #0055CC 100%) !important;
    color: #040A12 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    padding: 1rem 2rem !important;
    border: none !important;
    border-radius: 14px !important;
    margin-top: 1rem;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 30px rgba(0,200,255,0.2) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px rgba(0,200,255,0.35) !important;
}

/* ── Result Panel ── */
.result-box {
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.6s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-critical {
    background: linear-gradient(135deg, rgba(255,60,60,0.12), rgba(200,0,80,0.06));
    border: 1px solid rgba(255,60,60,0.35);
    box-shadow: 0 0 50px rgba(255,60,60,0.1);
}
.result-caution {
    background: linear-gradient(135deg, rgba(255,160,0,0.12), rgba(255,100,0,0.06));
    border: 1px solid rgba(255,160,0,0.35);
    box-shadow: 0 0 50px rgba(255,160,0,0.1);
}
.result-healthy {
    background: linear-gradient(135deg, rgba(0,200,255,0.1), rgba(0,100,200,0.06));
    border: 1px solid rgba(0,200,255,0.3);
    box-shadow: 0 0 50px rgba(0,200,255,0.08);
}
.result-icon {
    font-size: 3.2rem;
    margin-bottom: 0.8rem;
    display: block;
    animation: floatIcon 3s ease-in-out infinite;
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.result-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}
.result-critical .result-headline { color: #FF6B6B; }
.result-caution  .result-headline { color: #FFB347; }
.result-healthy  .result-headline { color: #00C8FF; }

.result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.8rem 0 0.2rem;
    letter-spacing: -0.02em;
}
.result-critical .result-value { color: #FF6B6B; }
.result-caution  .result-value { color: #FFB347; }
.result-healthy  .result-value { color: #00C8FF; }

.result-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: rgba(214,232,245,0.4);
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Level Gauge Bar ── */
.gauge-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 0.6rem 0 0.3rem;
    position: relative;
}
.gauge-fill {
    height: 100%;
    border-radius: 100px;
    animation: growGauge 1.2s cubic-bezier(0.4,0,0.2,1) both;
    position: relative;
}
@keyframes growGauge {
    from { width: 0% !important; }
}
.gauge-critical { background: linear-gradient(90deg, #FF3C3C, #FF6B6B); }
.gauge-caution  { background: linear-gradient(90deg, #FF8C00, #FFB347); }
.gauge-healthy  { background: linear-gradient(90deg, #0099CC, #00C8FF); }

/* ── Info Breakdown Card ── */
.breakdown-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.6rem;
    height: 100%;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.7s ease both;
}
.breakdown-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), transparent);
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.85rem;
}
.info-row:last-child { border-bottom: none; }
.info-row .lbl { color: rgba(214,232,245,0.45); }
.info-row .val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #D6E8F5;
    font-size: 0.82rem;
}

/* ── Status Badges ── */
.badge-row {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1rem;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 4px 11px;
    border-radius: 100px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-blue   { background: rgba(0,200,255,0.12);  color: #00C8FF;  border: 1px solid rgba(0,200,255,0.25); }
.badge-orange { background: rgba(255,160,0,0.12);  color: #FFB347;  border: 1px solid rgba(255,160,0,0.25); }
.badge-red    { background: rgba(255,60,60,0.12);  color: #FF6B6B;  border: 1px solid rgba(255,60,60,0.25); }
.badge-green  { background: rgba(0,220,130,0.12);  color: #00DC82;  border: 1px solid rgba(0,220,130,0.25); }

/* ── Divider ── */
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent); margin: 1.4rem 0; }

/* ── Reservoir Mini Cards ── */
.res-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.5rem;
}
.res-card {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 14px;
    padding: 0.9rem;
    text-align: center;
}
.res-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(214,232,245,0.35);
    margin-bottom: 0.3rem;
}
.res-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #00C8FF;
}
.res-unit { font-size: 0.6rem; color: rgba(214,232,245,0.3); }

/* Streamlit label override */
label { color: rgba(214,232,245,0.65) !important; }

/* ── st.metric dark theme ── */
[data-testid="stMetric"] {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 12px;
    padding: 0.7rem 0.9rem !important;
    margin-bottom: 0.5rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: rgba(214,232,245,0.38) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #00C8FF !important;
}
[data-testid="stMetricDelta"] { display: none; }

/* ── AUTH SECTION CSS ── */
.auth-container {
    max-width: 440px;
    margin: 4rem auto;
    padding: 2.5rem 2rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 24px;
    position: relative;
    overflow: hidden;
}
.auth-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00C8FF, transparent);
}
.auth-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #80E8FF 60%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.3rem;
}
.auth-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(214,232,245,0.35);
    text-align: center;
    margin-bottom: 1.8rem;
}
.auth-logo {
    text-align: center;
    font-size: 2.8rem;
    margin-bottom: 1rem;
}
/* Style text inputs inside auth forms */
.stTextInput input {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 12px !important;
    color: #D6E8F5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput input:focus {
    border-color: rgba(0,200,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,200,255,0.08) !important;
}
.stTextInput > label {
    color: rgba(214,232,245,0.55) !important;
    font-size: 0.82rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>

<!-- Cursor -->
<div class="cursor-dot" id="cursorDot"></div>
<div class="cursor-ring" id="cursorRing"></div>
<script>
const dot = document.getElementById('cursorDot');
const ring = document.getElementById('cursorRing');
let mx=0, my=0, rx=0, ry=0;
document.addEventListener('mousemove', e => { mx=e.clientX; my=e.clientY; dot.style.left=mx+'px'; dot.style.top=my+'px'; });
function animate(){
    rx += (mx-rx)*0.12; ry += (my-ry)*0.12;
    ring.style.left=rx+'px'; ring.style.top=ry+'px';
    requestAnimationFrame(animate);
}
animate();
document.querySelectorAll('button, a, input, select, [role=slider]').forEach(el=>{
    el.addEventListener('mouseenter',()=>{ring.style.width='52px';ring.style.height='52px';ring.style.borderColor='rgba(0,200,255,0.9)';});
    el.addEventListener('mouseleave',()=>{ring.style.width='32px';ring.style.height='32px';ring.style.borderColor='rgba(0,200,255,0.5)';});
});
</script>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION SECTION START
# ════════════════════════════════════════════════════════════════════════════

USERS_FILE = "users.csv"


def load_users() -> pd.DataFrame:
    """Load users from CSV. Creates the file with headers if it doesn't exist."""
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USERS_FILE, index=False)
        return df
    return pd.read_csv(USERS_FILE)


def save_user(username: str, password: str) -> None:
    """Append a new user row to users.csv."""
    df = load_users()
    new_row = pd.DataFrame([{"username": username, "password": password}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)


def username_exists(username: str) -> bool:
    """Return True if the username is already registered."""
    df = load_users()
    return username.strip().lower() in df["username"].str.strip().str.lower().values


def verify_credentials(username: str, password: str) -> bool:
    """Return True if username + password pair is valid."""
    df = load_users()
    df["username"] = df["username"].str.strip().str.lower()
    df["password"] = df["password"].str.strip()
    match = df[(df["username"] == username.strip().lower()) &
               (df["password"] == password.strip())]
    return not match.empty


def init_session_state() -> None:
    """Initialise all session-state keys used by the auth system."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"   # "login" | "signup"


def show_login_page() -> None:
    """Render the Login page."""
    st.markdown("""
    <div class="auth-container">
        <div class="auth-logo">🌊</div>
        <div class="auth-title">HydroSense</div>
        <div class="auth-subtitle">Sign in to continue</div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", key="login_password", placeholder="Enter your password", type="password")

        if st.button("🔑 Login", use_container_width=True, key="login_btn"):
            if not username.strip() or not password.strip():
                st.error("⚠️ Please fill in both username and password.")
            elif verify_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align:center;font-size:0.82rem;"
            "color:rgba(214,232,245,0.4);font-family:Space Grotesk,sans-serif;'>"
            "Don't have an account?</p>",
            unsafe_allow_html=True
        )

        if st.button("✨ Create Account", use_container_width=True, key="go_signup_btn"):
            st.session_state.auth_page = "signup"
            st.rerun()


def show_signup_page() -> None:
    """Render the Signup page."""
    st.markdown("""
    <div class="auth-container">
        <div class="auth-logo">🌊</div>
        <div class="auth-title">Create Account</div>
        <div class="auth-subtitle">Join HydroSense</div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        new_username = st.text_input("Choose Username", key="signup_username", placeholder="Pick a unique username")
        new_password = st.text_input("Create Password", key="signup_password", placeholder="Create a strong password", type="password")
        confirm_password = st.text_input("Confirm Password", key="signup_confirm", placeholder="Re-enter your password", type="password")

        if st.button("🚀 Sign Up", use_container_width=True, key="signup_btn"):
            if not new_username.strip() or not new_password.strip() or not confirm_password.strip():
                st.error("⚠️ All fields are required.")
            elif len(new_username.strip()) < 3:
                st.error("⚠️ Username must be at least 3 characters long.")
            elif len(new_password.strip()) < 4:
                st.error("⚠️ Password must be at least 4 characters long.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match. Please try again.")
            elif username_exists(new_username):
                st.error(f"❌ Username **{new_username}** is already taken. Please choose another.")
            else:
                save_user(new_username.strip(), new_password.strip())
                st.success(f"✅ Account created for **{new_username}**! You can now log in.")
                st.session_state.auth_page = "login"
                st.rerun()

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align:center;font-size:0.82rem;"
            "color:rgba(214,232,245,0.4);font-family:Space Grotesk,sans-serif;'>"
            "Already have an account?</p>",
            unsafe_allow_html=True
        )

        if st.button("← Back to Login", use_container_width=True, key="go_login_btn"):
            st.session_state.auth_page = "login"
            st.rerun()


def show_logout_button() -> None:
    """Render the logout button + welcome message inside the sidebar."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1rem 0 0.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                letter-spacing:0.18em;text-transform:uppercase;
                color:rgba(214,232,245,0.35);margin-bottom:4px;">Signed in as</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                font-weight:600;color:#00C8FF;">👤 {st.session_state.username}</div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(0,200,255,0.12);margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        if st.button("🔓 Logout", use_container_width=True, key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.auth_page = "login"
            st.rerun()


# ── Auth Gate ────────────────────────────────────────────────────────────────
init_session_state()

if not st.session_state.authenticated:
    if st.session_state.auth_page == "signup":
        show_signup_page()
    else:
        show_login_page()
    st.stop()   # <-- halts execution; nothing below renders until user is logged in

# ── Logout button available in sidebar for authenticated users ───────────────
show_logout_button()

# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION SECTION END
# ════════════════════════════════════════════════════════════════════════════


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = 'best_model.pkl'
    scaler_path = 'scaler.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Chennai Reservoir Intelligence System</div>
    <h1>HydroSense<br>Water Level Predictor</h1>
    <p>7-day ahead forecast for Chennai's four reservoirs using ensemble ML trained on 16 years of hydrological data.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-item"><div class="stat-val">2004–2020</div><div class="stat-lbl">Training Period</div></div>
    <div class="stat-item"><div class="stat-val">17</div><div class="stat-lbl">Feature Inputs</div></div>
    <div class="stat-item"><div class="stat-val">8</div><div class="stat-lbl">Models Compared</div></div>
    <div class="stat-item"><div class="stat-val">+7 Days</div><div class="stat-lbl">Forecast Horizon</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Reservoir Info Subheader ─────────────────────────────────────────────────
st.markdown("""
<div class="res-grid">
    <div class="res-card"><div class="res-name">Poondi</div><div class="res-val">93.46</div><div class="res-unit">MCft Capacity</div></div>
    <div class="res-card"><div class="res-name">Cholavaram</div><div class="res-val">31.00</div><div class="res-unit">MCft Capacity</div></div>
    <div class="res-card"><div class="res-name">Red Hills</div><div class="res-val">93.00</div><div class="res-unit">MCft Capacity</div></div>
    <div class="res-card"><div class="res-name">Chembarambakkam</div><div class="res-val">103.00</div><div class="res-unit">MCft Capacity</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Input Section ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

with col1:
    st.markdown('<div class="card"><div class="card-label">📅 Forecast Date</div>', unsafe_allow_html=True)
    forecast_date = st.date_input(
        "Prediction Target Date",
        value=date.today(),
        min_value=date(2004, 1, 1),
        help="Date for which you want to predict total reservoir level (7 days forward from inputs)"
    )
    lag_level = st.number_input(
        "Current Total Level (MCft)",
        min_value=0.0,
        max_value=10000.0,
        value=3200.0,
        step=50.0,
        help="Current total water level across all 4 reservoirs (from 12 days ago for lag feature)"
    )
    roll_level_std = st.number_input(
        "7-Day Level Std Dev (MCft)",
        min_value=0.0,
        max_value=2000.0,
        value=80.0,
        step=10.0,
        help="Standard deviation of total reservoir level over the past 7 days"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-label">🌧️ Reservoir Rainfall (mm)</div>', unsafe_allow_html=True)
    poondi_rain = st.slider("Poondi Rainfall (mm)", 0.0, 300.0, 20.0, step=0.5)
    cholavaram_rain = st.slider("Cholavaram Rainfall (mm)", 0.0, 300.0, 10.0, step=0.5)
    redhills_rain = st.slider("Red Hills Rainfall (mm)", 0.0, 300.0, 15.0, step=0.5)
    chembarambakkam_rain = st.slider("Chembarambakkam Rainfall (mm)", 0.0, 300.0, 12.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="card-label">📊 Rolling Averages</div>', unsafe_allow_html=True)
    total_rainfall = poondi_rain + cholavaram_rain + redhills_rain + chembarambakkam_rain
    st.markdown(f"""
    <div style="background:rgba(0,200,255,0.06);border:1px solid rgba(0,200,255,0.15);
        border-radius:12px;padding:0.9rem 1rem;margin-bottom:0.8rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
            color:rgba(214,232,245,0.4);letter-spacing:0.12em;text-transform:uppercase;
            margin-bottom:4px;">TOTAL DAILY RAINFALL</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
            font-weight:700;color:#00C8FF;">{total_rainfall:.1f} <span style="font-size:0.7rem;color:rgba(214,232,245,0.4);">mm</span></div>
    </div>
    """, unsafe_allow_html=True)
    roll_7_rain = st.number_input(
        "7-Day Rolling Avg Rainfall (mm)",
        min_value=0.0, max_value=200.0, value=18.0, step=1.0,
        help="Average of total daily rainfall over the past 7 days"
    )
    roll_30_rain = st.number_input(
        "30-Day Rolling Avg Rainfall (mm)",
        min_value=0.0, max_value=150.0, value=12.0, step=1.0,
        help="Average of total daily rainfall over the past 30 days"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict Button ───────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("🌊 Run 7-Day Forecast", use_container_width=True)

# ─── Prediction Logic ─────────────────────────────────────────────────────────
if predict_btn:
    d = pd.Timestamp(forecast_date)

    # Build feature vector (17 features — exact order from notebook)
    features = [
        poondi_rain, cholavaram_rain, redhills_rain, chembarambakkam_rain,  # 4
        d.year, d.month, d.quarter, d.dayofyear,                            # 4
        np.sin(2 * np.pi * d.month / 12),                                   # Month_sin
        np.cos(2 * np.pi * d.month / 12),                                   # Month_cos
        np.sin(2 * np.pi * d.dayofyear / 365),                              # Day_sin
        np.cos(2 * np.pi * d.dayofyear / 365),                              # Day_cos
        total_rainfall,                                                       # TOTAL_RAINFALL
        lag_level,                                                            # Lag_level
        roll_7_rain,                                                          # Roll_7_rain_mean
        roll_30_rain,                                                         # Roll_30_rain_mean
        roll_level_std                                                        # Roll_level_std
    ]

    X_input = np.array(features).reshape(1, -1)

    if model is None or scaler is None:
        # ── Demo Mode (no model files) ──────────────────────────────────────
        # Compute a plausible synthetic prediction for demo
        seasonal_boost = 200 * np.sin(2 * np.pi * d.month / 12 - 1.5) + 200
        predicted_level = round(lag_level * 0.97 + total_rainfall * 8 + seasonal_boost + np.random.normal(0, 50), 2)
        model_name = "Demo Mode (load best_model.pkl for live predictions)"
    else:
        X_scaled    = scaler.transform(X_input)
        predicted_level = round(float(model.predict(X_scaled)[0]), 2)
        model_name  = type(model).__name__

    # ── Capacity thresholds (total across 4 reservoirs: ~320 MCft) ──────────
    MAX_CAPACITY = 320.46
    pct_full     = min(100.0, round(predicted_level / MAX_CAPACITY * 100, 1))
    pct_full     = max(0.0, pct_full)

    if pct_full >= 75:
        status_cls  = "result-healthy"
        gauge_cls   = "gauge-healthy"
        icon        = "💧"
        headline    = "Healthy Reservoir Level"
        status_text = "Reservoirs are well-stocked. Adequate water supply expected."
        status_badge_cls = "badge-blue"
        status_badge_txt = "HEALTHY"
    elif pct_full >= 40:
        status_cls  = "result-caution"
        gauge_cls   = "gauge-caution"
        icon        = "⚠️"
        headline    = "Moderate Level — Monitor Closely"
        status_text = "Reservoir levels are below optimal. Conservation advised."
        status_badge_cls = "badge-orange"
        status_badge_txt = "CAUTION"
    else:
        status_cls  = "result-critical"
        gauge_cls   = "gauge-critical"
        icon        = "🚨"
        headline    = "Critical Low Level"
        status_text = "Reservoir levels critically low. Immediate action may be required."
        status_badge_cls = "badge-red"
        status_badge_txt = "CRITICAL"

    # Contextual badges
    badges = [(status_badge_cls, status_badge_txt)]
    if total_rainfall > 80:  badges.append(("badge-blue",   "HIGH RAINFALL"))
    elif total_rainfall > 30: badges.append(("badge-green", "MODERATE RAIN"))
    else:                     badges.append(("badge-orange", "LOW RAIN"))
    if d.month in [10, 11, 12]: badges.append(("badge-blue", "NE MONSOON"))
    elif d.month in [6, 7, 8]:  badges.append(("badge-blue", "SW MONSOON"))
    if lag_level > 250: badges.append(("badge-green", "GOOD INTAKE"))

    badges_html = "".join([f'<span class="badge {c}">{t}</span>' for c, t in badges[:5]])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1], gap="large")

    with r1:
        st.markdown(f"""
        <div class="result-box {status_cls}">
            <span class="result-icon">{icon}</span>
            <div class="result-headline">{headline}</div>
            <div class="result-value">{predicted_level:,.1f}</div>
            <div class="result-unit">MCft — Predicted Total Level</div>
            <div style="font-size:0.82rem;color:rgba(214,232,245,0.45);margin-bottom:10px;">
                {pct_full}% of total capacity
            </div>
            <div class="gauge-wrap">
                <div class="gauge-fill {gauge_cls}" style="width:{pct_full}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:rgba(214,232,245,0.3);margin-top:5px;margin-bottom:1rem;">
                <span>0</span><span>160 MCft</span><span>320 MCft</span>
            </div>
            <div class="badge-row">{badges_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        # Header
        st.markdown('<div class="card-label">📋 Forecast Breakdown</div>', unsafe_allow_html=True)

        # Metric table using native st.metric (styled via CSS already injected)
        ma, mb = st.columns(2)
        ma.metric("📅 Forecast Date",    d.strftime('%d %b %Y'))
        mb.metric("🌊 Predicted Level",  f"{predicted_level:,.1f} MCft")
        mc, md = st.columns(2)
        mc.metric("📊 Capacity Fill",    f"{pct_full}%")
        md.metric("🌧️ Total Rainfall",   f"{total_rainfall:.1f} mm")
        me, mf = st.columns(2)
        me.metric("💦 Lag Level",        f"{lag_level:,.0f} MCft")
        mf.metric("🤖 Model",            model_name.replace("Regressor","").strip())
        mg, mh = st.columns(2)
        mg.metric("📈 7-Day Rain Avg",   f"{roll_7_rain:.1f} mm/d")
        mh.metric("📆 Season",           f"Q{d.quarter} · {month_names[d.month-1]}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Individual reservoir rainfall as a small table
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;"
            "color:rgba(214,232,245,0.35);letter-spacing:0.12em;text-transform:uppercase;"
            "margin-bottom:0.4rem;'>Individual Reservoir Rainfall</p>",
            unsafe_allow_html=True
        )
        ra, rb, rc, rd = st.columns(4)
        ra.metric("Poondi",        f"{poondi_rain:.0f} mm")
        rb.metric("Cholavaram",    f"{cholavaram_rain:.0f} mm")
        rc.metric("Red Hills",     f"{redhills_rain:.0f} mm")
        rd.metric("Chemb.",        f"{chembarambakkam_rain:.0f} mm")

        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;"
            "color:rgba(214,232,245,0.25);text-align:center;margin-top:0.8rem;'>"
            "⚠ AI forecast · Add best_model.pkl + scaler.pkl for live predictions</p>",
            unsafe_allow_html=True
        )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
    border-top:1px solid rgba(255,255,255,0.05);
    font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    color:rgba(214,232,245,0.2);letter-spacing:0.1em;">
    HYDROSENSE v1.0 · Chennai Reservoir ML Project · Trained on 2004–2020 CWRD Data
</div>
""", unsafe_allow_html=True)
