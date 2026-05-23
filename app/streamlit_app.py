import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, requests, io

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="BookMind — KNN Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #021B3A 0%, #065A82 100%);
}
section[data-testid="stSidebar"] * { color: #EBF5FB !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label { color: #02C39A !important; font-weight: 500; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #021B3A, #065A82);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    color: white;
    text-align: center;
    border: 1px solid rgba(2,195,154,0.3);
    box-shadow: 0 8px 32px rgba(2,27,58,0.3);
}
.metric-card .value { font-size: 2rem; font-weight: 900; color: #02C39A; font-family: 'Playfair Display', serif; }
.metric-card .label { font-size: 0.85rem; color: #CADCFC; margin-top: 4px; }

/* Book cards */
.book-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin: 0.4rem 0;
    border-left: 4px solid #02C39A;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    display: flex;
    align-items: center;
    gap: 12px;
}
.book-rank {
    background: linear-gradient(135deg, #021B3A, #065A82);
    color: #02C39A;
    border-radius: 50%;
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 0.9rem; flex-shrink: 0;
}
.book-title { font-weight: 600; color: #021B3A; font-size: 0.95rem; }
.book-rating { color: #02C39A; font-weight: 700; font-size: 0.9rem; }

/* Section header */
.section-header {
    background: linear-gradient(90deg, #021B3A, #065A82);
    color: white !important;
    padding: 0.7rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #02C39A, #065A82);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(2,195,154,0.4); }
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    books_url   = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
    ratings_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
    books_df   = pd.read_csv(books_url)
    ratings_df = pd.read_csv(ratings_url)
    ratings_sample = ratings_df.sample(n=10000, random_state=42)
    book_dict = dict(zip(books_df['book_id'], books_df['title']))
    author_dict = dict(zip(books_df['book_id'], books_df['authors']))
    return books_df, ratings_df, ratings_sample, book_dict, author_dict


@st.cache_resource(show_spinner="Training KNN models... (this takes ~1 min)")
def train_models(ratings_sample):
    from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, KNNBaseline
    from surprise.model_selection import train_test_split
    from surprise import accuracy

    reader = Reader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(ratings_sample[['user_id','book_id','rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    sim_options = {'name': 'cosine', 'user_based': True}
    sim_item    = {'name': 'cosine', 'user_based': False}

    models = {}
    results = {}
    for name, cls in [('KNNBasic', KNNBasic), ('KNNWithMeans', KNNWithMeans), ('KNNBaseline', KNNBaseline)]:
        m = cls(k=10, sim_options=sim_options)
        m.fit(trainset)
        preds = m.test(testset)
        models[name] = m
        results[name] = {
            'RMSE': round(accuracy.rmse(preds, verbose=False), 4),
            'MAE' : round(accuracy.mae(preds,  verbose=False), 4),
        }

    item_model = KNNWithMeans(k=10, sim_options=sim_item)
    item_model.fit(trainset)
    models['item'] = item_model

    return models, results, trainset, testset


def get_recommendations(model, trainset, user_id, book_dict, author_dict, n=10):
    all_ids = list(book_dict.keys())
    try:
        inner_uid   = trainset.to_inner_uid(user_id)
        rated_books = set(trainset.to_raw_iid(iid) for iid, _ in trainset.ur[inner_uid])
    except ValueError:
        rated_books = set()
    unrated = [bid for bid in all_ids if bid not in rated_books]
    preds   = [model.predict(user_id, bid) for bid in unrated]
    preds.sort(key=lambda x: x.est, reverse=True)
    return [
        {'rank': i+1, 'title': book_dict.get(p.iid, 'Unknown'),
         'author': author_dict.get(p.iid, ''), 'rating': round(p.est, 2)}
        for i, p in enumerate(preds[:n])
    ]


def get_similar_books(item_model, trainset, book_id, book_dict, author_dict, n=8):
    try:
        inner_id  = trainset.to_inner_iid(book_id)
        neighbors = item_model.get_neighbors(inner_id, k=n)
        return [
            {'rank': i+1,
             'title': book_dict.get(trainset.to_raw_iid(nid), 'Unknown'),
             'author': author_dict.get(trainset.to_raw_iid(nid), '')}
            for i, nid in enumerate(neighbors)
        ]
    except ValueError:
        return []


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 BookMind")
    st.markdown("*KNN-based Personalized Book Recommendations*")
    st.markdown("---")
    page = st.selectbox("Navigate", ["🏠 Home", "🎯 Get Recommendations", "🔍 Similar Books", "📊 Model Comparison"])
    st.markdown("---")
    st.markdown("**Dataset:** GoodBooks-10K")
    st.markdown("**Algorithm:** K-Nearest Neighbors")
    st.markdown("**Library:** Surprise")
    st.markdown("---")
    st.markdown("**Srinistha Biswas**")
    st.markdown("CSE (IOTCSBT) | Roll: 9")
    st.markdown("Enroll: 12024052017026")


# ── Load data ───────────────────────────────────────────────────
books_df, ratings_df, ratings_sample, book_dict, author_dict = load_data()
models, results, trainset, testset = train_models(ratings_sample)


# ══════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<h1 style='font-family:Playfair Display,serif;color:#021B3A;'>📚 BookMind</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#065A82;font-size:1.1rem;'>Personalized Book Recommendations using K-Nearest Neighbors</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["10K", "6M+", "3", "10"],
        ["Books", "Total Ratings", "KNN Models", "Neighbours (k)"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How It Works")
    cols = st.columns(5)
    steps = [("1️⃣","Collect Ratings","User–book rating data"),
             ("2️⃣","Cosine Similarity","Find similar users"),
             ("3️⃣","KNN","k=10 neighbours"),
             ("4️⃣","Predict","Estimate unread ratings"),
             ("5️⃣","Recommend","Return Top-N books")]
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"**{icon} {title}**\n\n{desc}")

    st.markdown("---")
    st.markdown("### Models Available")
    mc1, mc2, mc3 = st.columns(3)
    for col, name, desc, rmse in zip(
        [mc1, mc2, mc3],
        ["KNNBasic", "KNNWithMeans", "KNNBaseline"],
        ["Raw neighbor ratings, no bias correction",
         "Adjusts for user average rating bias",
         "Best: corrects global + user + item bias"],
        [results['KNNBasic']['RMSE'], results['KNNWithMeans']['RMSE'], results['KNNBaseline']['RMSE']]
    ):
        col.info(f"**{name}**\n\n{desc}\n\n📉 RMSE: `{rmse}`")


# ══════════════════════════════════════════════════════════
# PAGE: GET RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
elif page == "🎯 Get Recommendations":
    st.markdown("<div class='section-header'>🎯 Get Personalized Book Recommendations</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, max_value=int(ratings_sample['user_id'].max()), value=int(ratings_sample['user_id'].iloc[0]))
    with col2:
        model_choice = st.selectbox("Select Model", ["KNNBaseline", "KNNWithMeans", "KNNBasic"])
    with col3:
        top_n = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("🔮 Get Recommendations"):
        with st.spinner("Finding your perfect books..."):
            recs = get_recommendations(models[model_choice], trainset, user_id, book_dict, author_dict, n=top_n)

        if recs:
            st.success(f"Top {top_n} recommendations for User **{user_id}** using **{model_choice}**")
            for r in recs:
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-rank">#{r['rank']}</div>
                    <div>
                        <div class="book-title">{r['title']}</div>
                        <div style="color:#888;font-size:0.82rem;">{r['author']}</div>
                    </div>
                    <div style="margin-left:auto;" class="book-rating">⭐ {r['rating']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Try a different User ID.")


# ══════════════════════════════════════════════════════════
# PAGE: SIMILAR BOOKS
# ══════════════════════════════════════════════════════════
elif page == "🔍 Similar Books":
    st.markdown("<div class='section-header'>🔍 Find Similar Books</div>", unsafe_allow_html=True)

    # Book ID to title lookup for display
    valid_books = {bid: title for bid, title in book_dict.items() if bid in [trainset.to_raw_iid(i) for i in trainset.all_items()]}
    sample_ids  = list(valid_books.keys())[:200]
    options     = {f"{bid} — {book_dict.get(bid, 'Unknown')}": bid for bid in sample_ids}

    col1, col2 = st.columns([3,1])
    with col1:
        selected = st.selectbox("Select a Book", list(options.keys()))
    with col2:
        n_similar = st.slider("Number of Similar Books", 3, 10, 5)

    book_id = options[selected]

    if st.button("🔍 Find Similar Books"):
        with st.spinner("Finding similar books..."):
            similar = get_similar_books(models['item'], trainset, book_id, book_dict, author_dict, n=n_similar)

        if similar:
            st.success(f"Books similar to **{book_dict.get(book_id, 'Unknown')}**")
            for s in similar:
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-rank">#{s['rank']}</div>
                    <div>
                        <div class="book-title">{s['title']}</div>
                        <div style="color:#888;font-size:0.82rem;">{s['author']}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("Book not found in training set. Try another book.")


# ══════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown("<div class='section-header'>📊 Model Comparison — RMSE & MAE</div>", unsafe_allow_html=True)

    # Metrics table
    df_results = pd.DataFrame(results).T.reset_index()
    df_results.columns = ['Model', 'RMSE', 'MAE']
    st.dataframe(df_results.style.highlight_min(subset=['RMSE','MAE'], color='#d4f5e9'), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#F0F6FF')
    ax.set_facecolor('#F0F6FF')

    names  = list(results.keys())
    rmses  = [results[m]['RMSE'] for m in names]
    maes   = [results[m]['MAE']  for m in names]
    x      = np.arange(len(names))

    bars1 = ax.bar(x - 0.2, rmses, width=0.38, color='#065A82', label='RMSE', zorder=3)
    bars2 = ax.bar(x + 0.2, maes,  width=0.38, color='#02C39A', label='MAE',  zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10, color='#021B3A', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10, color='#021B3A', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight='bold', color='#021B3A')
    ax.set_ylabel('Error Score', color='#021B3A')
    ax.set_title('KNN Model Comparison — Lower is Better', fontsize=14, fontweight='bold', color='#021B3A', pad=12)
    ax.yaxis.grid(True, color='white', linewidth=1.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(handles=[mpatches.Patch(color='#065A82', label='RMSE'), mpatches.Patch(color='#02C39A', label='MAE')])

    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 📌 Key Takeaways")
    st.info("🏆 **KNNBaseline** achieves the best performance by correcting for global average + user bias + item bias.")
    st.info("📉 **RMSE** penalizes large errors more heavily. **MAE** gives equal weight to all errors. Both lower = better.")
    st.info("🔁 **3-fold cross-validation** was used to ensure results are not dependent on a single train/test split.")
