import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.predict import predict


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EstateIQ | Property Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "cleaned_house_prices.csv"
MODEL_RESULTS_PATH = ROOT / "data" / "processed" / "model_comparison.csv"


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #0B1220;
        --panel: #111C2E;
        --panel-2: #16243A;
        --teal: #2DD4BF;
        --teal-dark: #0F766E;
        --text: #E8F0FA;
        --muted: #91A3B9;
        --line: rgba(148, 163, 184, 0.18);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(20, 104, 112, 0.16), transparent 30%),
            #0B1220;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #080F1C;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #F4F8FC !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #D5E0ED;
    }

    .brand-mark {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .hero {
        background: linear-gradient(115deg, #14243B 0%, #123B4A 58%, #11645F 100%);
        border: 1px solid rgba(94, 234, 212, 0.18);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 28px;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.20);
    }

    .hero h1 {
        font-size: clamp(2rem, 3vw, 2.8rem) !important;
        margin: 0 0 0.45rem 0;
    }

    .hero p {
        color: #C5D8E6;
        font-size: 1rem;
        margin: 0.2rem 0;
    }

    .eyebrow {
        color: #5EEAD4;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .section-heading {
        margin: 1.4rem 0 0.2rem 0;
    }

    .section-subtitle {
        color: #91A3B9;
        margin-top: 0;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
    }

    div[data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(17, 28, 46, 0.98), rgba(19, 37, 56, 0.92));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px 24px 18px 24px;
    }

    div[data-testid="stForm"] label {
        color: #C9D7E7 !important;
        font-weight: 600;
        font-size: 0.86rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #0D1828 !important;
        border: 1px solid #2A3B52 !important;
        border-radius: 9px !important;
        color: #F1F5F9 !important;
    }

    input, textarea {
        color: #F1F5F9 !important;
    }

    div[data-baseweb="select"] span {
        color: #F1F5F9 !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(100deg, #0F766E, #14B8A6) !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.8rem;
        box-shadow: 0 8px 22px rgba(20, 184, 166, 0.16);
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        border: 1px solid #5EEAD4 !important;
        background: linear-gradient(100deg, #115E59, #0F766E) !important;
    }

    .metric-card {
        background: linear-gradient(145deg, #142238, #101B2C);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 112px;
    }

    .metric-label {
        color: #91A3B9;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-value {
        color: #F3F8FC;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .result-card {
        background: linear-gradient(120deg, #0F3D3B, #124B4A 60%, #145B58);
        border: 1px solid rgba(94, 234, 212, 0.38);
        border-radius: 20px;
        padding: 27px 30px;
        margin-top: 1.3rem;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
    }

    .result-label {
        color: #A7F3D0;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .result-price {
        color: #FFFFFF;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.1rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.2;
        margin: 0.55rem 0 0.2rem 0;
    }

    .result-exact {
        color: #D1FAE5;
        font-size: 0.92rem;
    }

    .info-panel {
        background: #111C2E;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 18px 20px;
    }

    .muted {
        color: #91A3B9 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        background: #111C2E;
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    hr {
        border-color: var(--line) !important;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 800px) {
        .block-container { padding: 1rem 1rem 2rem 1rem; }
        .hero { padding: 23px 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def format_price(price):
    price = float(price)
    if price >= 10_000_000:
        return f"₹{price / 10_000_000:.2f} Cr"
    if price >= 100_000:
        return f"₹{price / 100_000:.2f} Lakh"
    return f"₹{price:,.0f}"


def load_clean_data():
    if CLEAN_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DATA_PATH)
    return None


def get_options(data, column, fallback):
    if data is not None and column in data.columns:
        values = (
            data[column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        values = sorted(v for v in values.unique().tolist() if v and v.lower() != "nan")
        if values:
            return values
    return fallback


def metric_card(label, value, note=""):
    note_html = f'<div class="muted" style="font-size:0.78rem;margin-top:5px;">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_theme(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#D5E0ED"),
        title_font=dict(size=17, color="#F4F8FC"),
        height=height,
        margin=dict(l=15, r=15, t=55, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


# ============================================================
# LOAD DATA AND NAVIGATION
# ============================================================

market_data = load_clean_data()

st.sidebar.markdown(
    """
    <div style="padding: 8px 4px 18px 4px;">
        <div style="font-size:1.65rem;font-weight:800;color:#F4F8FC;font-family:Manrope,sans-serif;">🏡 EstateIQ</div>
        <div style="color:#91A3B9;font-size:0.78rem;margin-top:3px;">PROPERTY INTELLIGENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Price Predictor", "Market Analytics", "Model Insights", "About Project"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="color:#91A3B9;font-size:0.75rem;line-height:1.7;">
        <b style="color:#D5E0ED;">ESTATEIQ</b><br>
        Indian residential property listing intelligence.<br><br>
        <span style="color:#5EEAD4;">●</span> ML prediction engine
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">INDIA REAL ESTATE • DATA INTELLIGENCE</div>
        <h1>🏡 EstateIQ</h1>
        <p style="font-size:1.08rem;font-weight:600;color:#E8F7F5;">
            Understand property value. Make more informed decisions.
        </p>
        <p>Explore residential listing trends and estimate property prices with machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<div class="eyebrow">PLATFORM OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("## Real estate intelligence at a glance")
    st.markdown(
        '<p class="section-subtitle">A data-led view of Indian residential property listings and estimated values.</p>',
        unsafe_allow_html=True,
    )

    if market_data is not None and "price_rupees" in market_data.columns:
        total_listings = len(market_data)
        median_price = market_data["price_rupees"].median()
        avg_price = market_data["price_rupees"].mean()
        locations_count = market_data["location"].nunique() if "location" in market_data.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Listings analysed", f"{total_listings:,}", "Cleaned dataset")
        with c2:
            metric_card("Median listing price", format_price(median_price), "Across available listings")
        with c3:
            metric_card("Average listing price", format_price(avg_price), "Across available listings")
        with c4:
            metric_card("Locations covered", f"{locations_count:,}", "Distinct dataset locations")

        st.markdown("### Market snapshot")
        left, right = st.columns([1.15, 0.85])

        with left:
            if "location" in market_data.columns:
                city_data = (
                    market_data.groupby("location", as_index=False)["price_rupees"]
                    .median()
                    .sort_values("price_rupees", ascending=False)
                    .head(10)
                )
                fig = px.bar(
                    city_data.sort_values("price_rupees"),
                    x="price_rupees",
                    y="location",
                    orientation="h",
                    title="Top 10 locations by median listing price",
                    labels={"price_rupees": "Median listing price (₹)", "location": "Location"},
                    color="price_rupees",
                    color_continuous_scale=["#155E75", "#2DD4BF"],
                )
                st.plotly_chart(chart_theme(fig, 390), use_container_width=True)

        with right:
            fig = px.histogram(
                market_data,
                x="price_rupees",
                nbins=45,
                title="Property listing price distribution",
                labels={"price_rupees": "Listing price (₹)"},
                color_discrete_sequence=["#2DD4BF"],
            )
            st.plotly_chart(chart_theme(fig, 390), use_container_width=True)

        st.caption(
            "Dataset figures describe observed asking/listing prices, not verified sale transactions. "
            "The dataset is observational and may not represent the entire Indian housing market."
        )
    else:
        st.warning(
            "The cleaned dataset was not found at data/processed/cleaned_house_prices.csv. "
            "The prediction page can still work, but overview analytics need this file."
        )

    st.markdown("### Explore EstateIQ")
    a, b, c = st.columns(3)
    with a:
        st.markdown("""<div class="info-panel"><b>Price Predictor</b><p class="muted">Estimate a property's listing price using the trained machine-learning model.</p></div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class="info-panel"><b>Market Analytics</b><p class="muted">Explore location-level pricing patterns and property characteristics.</p></div>""", unsafe_allow_html=True)
    with c:
        st.markdown("""<div class="info-panel"><b>Model Insights</b><p class="muted">Compare regression models using evaluation metrics.</p></div>""", unsafe_allow_html=True)


# ============================================================
# PRICE PREDICTOR
# ============================================================

elif page == "Price Predictor":
    st.markdown('<div class="eyebrow">MACHINE LEARNING TOOL</div>', unsafe_allow_html=True)
    st.markdown("## Property price estimator")
    st.markdown(
        '<p class="section-subtitle">Enter the property details below to estimate its residential listing price.</p>',
        unsafe_allow_html=True,
    )

    location_options = get_options(
        market_data, "location",
        ["Pune", "Mumbai", "New Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"],
    )
    transaction_options = get_options(market_data, "transaction", ["Resale", "New Property"])
    furnishing_options = get_options(market_data, "furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
    facing_options = get_options(market_data, "facing", ["East", "West", "North", "South", "Unknown"])
    overlooking_options = get_options(market_data, "overlooking", ["Unknown"])
    ownership_options = get_options(market_data, "ownership", ["Freehold", "Leasehold", "Unknown"])
    parking_options = get_options(market_data, "parking_type", ["Unknown", "Covered", "Open"])

    def preferred_index(options, preferred):
        return options.index(preferred) if preferred in options else 0

    with st.form("estateiq_property_form"):
        st.markdown("#### 📍 Location & property configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            location = st.selectbox(
                "Property location",
                location_options,
                index=preferred_index(location_options, "Pune"),
            )
            bhk = st.number_input("BHK", min_value=1, max_value=20, value=2, step=1)

        with col2:
            transaction = st.selectbox(
                "Transaction type",
                transaction_options,
                index=preferred_index(transaction_options, "Resale"),
            )
            bathroom = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)

        with col3:
            furnishing = st.selectbox(
                "Furnishing",
                furnishing_options,
                index=preferred_index(furnishing_options, "Semi-Furnished"),
            )
            balcony = st.number_input("Balconies", min_value=0, max_value=20, value=1, step=1)

        st.markdown("---")
        st.markdown("#### 📐 Area & building details")
        col4, col5, col6 = st.columns(3)

        with col4:
            carpet_area = st.number_input(
                "Carpet area (sq. ft.)", min_value=50, max_value=50000, value=1000, step=50
            )
            super_area = st.number_input(
                "Super area (sq. ft.)", min_value=0, max_value=100000, value=1200, step=50
            )

        with col5:
            floor_number = st.number_input(
                "Floor number", min_value=0, max_value=100, value=2, step=1
            )
            total_floors = st.number_input(
                "Total floors", min_value=1, max_value=150, value=10, step=1
            )

        with col6:
            parking_count = st.number_input(
                "Parking count", min_value=0, max_value=20, value=1, step=1
            )
            parking_type = st.selectbox("Parking type", parking_options)

        st.markdown("---")
        st.markdown("#### 🧭 Additional property features")
        col7, col8, col9 = st.columns(3)

        with col7:
            facing = st.selectbox("Facing", facing_options)
        with col8:
            overlooking = st.selectbox("Overlooking", overlooking_options)
        with col9:
            ownership = st.selectbox("Ownership", ownership_options)

        submitted = st.form_submit_button(
            "✦  Estimate property listing price",
            use_container_width=True,
        )

    if submitted:
        if total_floors < max(floor_number, 1):
            st.error("Total floors cannot be lower than the selected property floor.")
        elif super_area > 0 and super_area < carpet_area:
            st.error("Super area should not be smaller than carpet area.")
        else:
            property_data = {
                "location": location,
                "transaction": transaction,
                "furnishing": furnishing,
                "facing": facing,
                "overlooking": overlooking,
                "ownership": ownership,
                "parking_type": parking_type,
                "bathroom": float(bathroom),
                "balcony": float(balcony),
                "carpet_area_sqft": float(carpet_area),
                "super_area_sqft": float(super_area) if super_area > 0 else None,
                "floor_number": float(floor_number),
                "total_floors": float(total_floors),
                "bhk": float(bhk),
                "parking_count": float(parking_count),
            }

            try:
                with st.spinner("Analysing property details..."):
                    estimated_price = predict(property_data)

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Estimated property listing price</div>
                        <div class="result-price">{format_price(estimated_price)}</div>
                        <div class="result-exact">Estimated value: ₹{estimated_price:,.0f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.success("Your property estimate is ready.")
                st.caption(
                    "This is a machine-learning estimate based on historical property listings. "
                    "It is not a formal valuation or a guarantee of the final sale price."
                )

                summary = pd.DataFrame(
                    {"Property feature": list(property_data.keys()),
                     "Selected value": list(property_data.values())}
                )
                with st.expander("View property input summary"):
                    st.dataframe(summary, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Prediction error: {exc}")


# ============================================================
# MARKET ANALYTICS
# ============================================================

elif page == "Market Analytics":
    st.markdown('<div class="eyebrow">INDIAN PROPERTY MARKET</div>', unsafe_allow_html=True)
    st.markdown("## Market analytics")
    st.markdown(
        '<p class="section-subtitle">Explore listing-price patterns and property mix in the available dataset.</p>',
        unsafe_allow_html=True,
    )

    if market_data is None:
        st.warning("Cleaned dataset not found. Add cleaned_house_prices.csv to data/processed to view analytics.")
    else:
        required = {"price_rupees"}
        if not required.issubset(market_data.columns):
            st.error("The cleaned dataset does not contain the required price_rupees column.")
        else:
            filtered = market_data.copy()

            if "location" in filtered.columns:
                locations = sorted(filtered["location"].dropna().astype(str).unique())
                selected_locations = st.multiselect(
                    "Filter locations",
                    locations,
                    default=locations[: min(8, len(locations))],
                )
                if selected_locations:
                    filtered = filtered[filtered["location"].astype(str).isin(selected_locations)]

            if filtered.empty:
                st.info("No listings match the selected filters.")
            else:
                m1, m2, m3 = st.columns(3)
                with m1:
                    metric_card("Listings in view", f"{len(filtered):,}")
                with m2:
                    metric_card("Median listing price", format_price(filtered["price_rupees"].median()))
                with m3:
                    metric_card("Average listing price", format_price(filtered["price_rupees"].mean()))

                left, right = st.columns(2)

                with left:
                    if "bhk" in filtered.columns:
                        bhk_data = (
                            filtered.groupby("bhk", as_index=False)["price_rupees"]
                            .median()
                            .sort_values("bhk")
                        )
                        fig = px.bar(
                            bhk_data,
                            x="bhk",
                            y="price_rupees",
                            title="Median listing price by BHK",
                            labels={"bhk": "BHK", "price_rupees": "Median listing price (₹)"},
                            color_discrete_sequence=["#2DD4BF"],
                        )
                        st.plotly_chart(chart_theme(fig), use_container_width=True)

                with right:
                    if "furnishing" in filtered.columns:
                        furnishing_data = (
                            filtered.groupby("furnishing", as_index=False)["price_rupees"]
                            .median()
                            .sort_values("price_rupees", ascending=False)
                        )
                        fig = px.bar(
                            furnishing_data,
                            x="furnishing",
                            y="price_rupees",
                            title="Median listing price by furnishing",
                            labels={"furnishing": "Furnishing", "price_rupees": "Median listing price (₹)"},
                            color_discrete_sequence=["#14B8A6"],
                        )
                        st.plotly_chart(chart_theme(fig), use_container_width=True)

                left2, right2 = st.columns(2)

                with left2:
                    if {"carpet_area_sqft", "price_rupees"}.issubset(filtered.columns):
                        plot_data = filtered[
                            (filtered["carpet_area_sqft"] > 0)
                            & (filtered["price_rupees"] > 0)
                        ].sample(min(5000, len(filtered)), random_state=42)
                        fig = px.scatter(
                            plot_data,
                            x="carpet_area_sqft",
                            y="price_rupees",
                            title="Carpet area vs listing price",
                            labels={
                                "carpet_area_sqft": "Carpet area (sq. ft.)",
                                "price_rupees": "Listing price (₹)",
                            },
                            opacity=0.45,
                            color_discrete_sequence=["#5EEAD4"],
                        )
                        st.plotly_chart(chart_theme(fig), use_container_width=True)

                with right2:
                    if "location" in filtered.columns:
                        city_data = (
                            filtered.groupby("location", as_index=False)
                            .agg(
                                median_price=("price_rupees", "median"),
                                listings=("price_rupees", "size"),
                            )
                            .sort_values("median_price", ascending=False)
                            .head(12)
                        )
                        fig = px.bar(
                            city_data.sort_values("median_price"),
                            x="median_price",
                            y="location",
                            orientation="h",
                            title="Location comparison by median price",
                            labels={"median_price": "Median listing price (₹)", "location": "Location"},
                            color="median_price",
                            color_continuous_scale=["#155E75", "#2DD4BF"],
                        )
                        st.plotly_chart(chart_theme(fig, 420), use_container_width=True)

                st.caption(
                    "All charts are descriptive summaries of the cleaned listing dataset. "
                    "They should not be interpreted as verified transaction-price statistics."
                )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":
    st.markdown('<div class="eyebrow">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.markdown("## Machine learning performance")
    st.markdown(
        '<p class="section-subtitle">Compare the regression models evaluated for EstateIQ.</p>',
        unsafe_allow_html=True,
    )

    if MODEL_RESULTS_PATH.exists():
        results = pd.read_csv(MODEL_RESULTS_PATH)

        if "MAE" not in results.columns or "Model" not in results.columns:
            st.error("The model comparison file is missing expected columns.")
        else:
            best = results.sort_values("MAE").iloc[0]
            r2_column = "R2 Score" if "R2 Score" in results.columns else None

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Lowest MAE model", str(best["Model"]))
            with c2:
                metric_card("Lowest MAE", format_price(best["MAE"]))
            with c3:
                metric_card("R² score", f'{best[r2_column]:.4f}' if r2_column else "N/A")

            st.markdown("### Evaluation results")
            display_results = results.copy()
            st.dataframe(display_results, use_container_width=True, hide_index=True)

            left, right = st.columns(2)
            with left:
                fig = px.bar(
                    results,
                    x="Model",
                    y="MAE",
                    title="Mean absolute error (lower is better)",
                    labels={"MAE": "MAE (₹)"},
                    color="Model",
                    color_discrete_sequence=["#2DD4BF", "#3B82F6", "#A78BFA"],
                )
                st.plotly_chart(chart_theme(fig), use_container_width=True)

            with right:
                log_r2_column = "Log R2 Score" if "Log R2 Score" in results.columns else None
                if log_r2_column:
                    fig = px.bar(
                        results,
                        x="Model",
                        y=log_r2_column,
                        title="R² score on log-price scale",
                        labels={log_r2_column: "Log-scale R²"},
                        color="Model",
                        color_discrete_sequence=["#2DD4BF", "#3B82F6", "#A78BFA"],
                    )
                    st.plotly_chart(chart_theme(fig), use_container_width=True)

            st.info(
                "The selected prediction engine is the Random Forest Regressor. "
                "Model metrics describe performance on the held-out test set; "
                "they do not guarantee the accuracy of an individual estimate."
            )
    else:
        st.warning("Model comparison report not found. Run the model training script first.")


# ============================================================
# ABOUT
# ============================================================

elif page == "About Project":
    st.markdown('<div class="eyebrow">PROJECT INFORMATION</div>', unsafe_allow_html=True)
    st.markdown("## About EstateIQ")
    st.markdown(
        """
        EstateIQ is a machine-learning-based property listing price estimation
        platform built using an Indian residential property listings dataset.

        ### Project objectives
        - Analyse residential property listing patterns across Indian locations.
        - Explore how property attributes relate to listing prices.
        - Estimate listing prices using supervised machine learning.
        - Compare regression algorithms using held-out evaluation data.

        ### Technology stack
        - **Programming:** Python
        - **Data analysis:** Pandas, NumPy
        - **Machine learning:** Scikit-learn, Random Forest, XGBoost
        - **Visualisation:** Plotly
        - **Application:** Streamlit

        ### Dataset
        House Price (India), by Juhi Bhojani, hosted on Kaggle.
        The dataset contains property listing information. Its prices should be
        interpreted as listing/asking prices rather than verified transaction prices.

        ### Model note
        Random Forest was selected for the current prediction workflow based on
        its performance on the original rupee-scale evaluation metrics. XGBoost
        performed slightly better on some log-scale metrics, so model performance
        depends on the metric being considered.
        """
    )

    st.markdown(
        """
        <div class="info-panel">
            <div class="eyebrow">IMPORTANT DISCLAIMER</div>
            <b>EstateIQ provides an indicative machine-learning estimate.</b>
            <p class="muted" style="margin-bottom:0;">
                It is not a certified property valuation, investment recommendation,
                or guarantee of a property's final selling price.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
                color:#71839A;font-size:0.76rem;padding:4px 2px 12px 2px;">
        <span>© EstateIQ • House Price Prediction & Property Intelligence</span>
        <span>Built with Python · Machine Learning · Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)
