import streamlit as st

# ---------------------------------------------------------------------
# Branding constants
# ---------------------------------------------------------------------
FOOTER = "Developed with Streamlit with 💗 by CE Team Innovation Lab 2025"
LOGO_URL = "https://raw.githubusercontent.com/skappal7/TextAnalyser/refs/heads/main/logo.png"


# ---------------------------------------------------------------------
# Branding & Layout
# ---------------------------------------------------------------------
def inject_branding():
    """Inject custom CSS for logo, background, and footer."""
    st.markdown(
        f"""
        <style>
        .top-right-logo {{
            position: fixed;
            top: 12px;
            right: 16px;
            z-index: 1000;
        }}
        .stApp {{
            background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
            color: #e2e8f0;
        }}
        .footer {{
            position: fixed;
            bottom: 8px;
            left: 0;
            right: 0;
            text-align: center;
            opacity: 0.85;
            font-size: 0.9rem;
            color: #94a3b8;
        }}
        h1, h2, h3, h4, h5 {{
            color: #f1f5f9 !important;
        }}
        .stButton>button {{
            border-radius: 8px;
            background-color: #2563eb !important;
            color: white !important;
            border: none;
            font-weight: 600;
            transition: 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #1d4ed8 !important;
            transform: scale(1.03);
        }}
        </style>

        <div class="top-right-logo">
            <img src="{LOGO_URL}" alt="Logo" width="96"/>
        </div>
        <div class="footer">{FOOTER}</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Section Header
# ---------------------------------------------------------------------
def section(title: str):
    """Stylized section heading."""
    st.markdown(
        f"<h3 style='margin-top:14px;color:#e2e8f0'>{title}</h3>",
        unsafe_allow_html=True,
    )
