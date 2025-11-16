# app.py
import os
import time
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

load_dotenv() 

# Use the modern OpenAI client pattern
try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("OpenAI package not found. Install it with: pip install openai") from e

# Optional PDF parsing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Configuration
MODEL = "gpt-4.1"
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
MAX_WORKERS = 6
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
CONFIDENCE_MIN = 0
CONFIDENCE_MAX = 100
TEXT_PREVIEW_LEN = 120
KEYWORDS_MAX = 10
VALID_SENTIMENTS = {"positive", "neutral", "negative"}
FILE_EXTENSIONS = {".txt", ".csv", ".json", ".pdf"}

# Sentiment colors
SENTIMENT_COLORS = {
    "positive": "#10b981",
    "neutral": "#6b7280",
    "negative": "#ef4444"
}

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'analyzed_texts' not in st.session_state:
    st.session_state.analyzed_texts = None

# Initialize OpenAI client
def get_client(api_key: str | None = None) -> OpenAI:
    """Initialize OpenAI client with provided or environment API key."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not provided. Set in sidebar or OPENAI_API_KEY env var.")
    return OpenAI(api_key=key)

def call_chat_completion(client, messages, model=MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS) -> str:
    """Call OpenAI chat completion with exponential backoff retry."""
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_exception = e
            if attempt >= MAX_RETRIES - 1:
                break
            wait_time = RETRY_BACKOFF * (2 ** attempt)
            time.sleep(wait_time)
    
    raise RuntimeError(f"Max retries exceeded. Last error: {last_exception}")

RESULT_SCHEMA_PROMPT = """You are a sentiment-analysis assistant. Given a text input, produce a JSON object only.
The JSON must contain the following keys:
- sentiment: one of "positive", "neutral", "negative"
- confidence: integer 0-100 (confidence percentage)
- keywords: array of 3-10 short keywords/phrases (strings)
- explanation: short plain-language explanation (1-3 sentences)
Return only valid JSON (no surrounding backticks or explanatory text)."""

def build_analysis_prompt(text: str) -> list:
    return [
        {"role": "system", "content": RESULT_SCHEMA_PROMPT},
        {"role": "user", "content": f"Analyze this text for sentiment and return a JSON:\n\n'''{text}'''\n\nRespond with JSON."}
    ]

def _parse_json_response(raw: str) -> dict:
    """Parse JSON from model response, with fallback extraction."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError, TypeError):
            return {}

def _sanitize_sentiment(sentiment: str) -> str:
    """Normalize sentiment to valid values."""
    normalized = sentiment.lower().strip()
    return normalized if normalized in VALID_SENTIMENTS else "neutral"

def _sanitize_confidence(confidence_val) -> int:
    """Normalize confidence to 0-100 range."""
    try:
        conf = int(float(confidence_val))
        return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, conf))
    except (ValueError, TypeError):
        return 50

def _sanitize_keywords(keywords) -> list:
    """Normalize keywords to list of strings."""
    if isinstance(keywords, str):
        return [k.strip() for k in keywords.split(",") if k.strip()]
    if isinstance(keywords, list):
        return keywords[:KEYWORDS_MAX]
    return []

def analyze_text(client, text: str) -> dict:
    """Analyze text sentiment and return structured result."""
    if not text or not text.strip():
        raise ValueError("Empty text provided.")

    messages = build_analysis_prompt(text)
    raw = call_chat_completion(client, messages)
    parsed = _parse_json_response(raw)

    if not parsed:
        return {
            "sentiment": "neutral",
            "confidence": 50,
            "keywords": [],
            "explanation": "Failed to parse model output.",
            "raw_output": raw
        }

    return {
        "sentiment": _sanitize_sentiment(parsed.get("sentiment", "")),
        "confidence": _sanitize_confidence(parsed.get("confidence", parsed.get("score"))),
        "keywords": _sanitize_keywords(parsed.get("keywords", [])),
        "explanation": parsed.get("explanation", "")
    }

def analyze_batch(client, texts: list, max_workers: int = MAX_WORKERS) -> list:
    """Analyze multiple texts in parallel."""
    results: list[dict | None] = [None] * len(texts)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_text, client, text): idx for idx, text in enumerate(texts)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "error": str(e),
                    "sentiment": "neutral",
                    "confidence": 0,
                    "keywords": [],
                    "explanation": ""
                }
    
    return [r if r is not None else {"sentiment": "neutral", "confidence": 0, "keywords": [], "explanation": "Error"} for r in results]

def read_text_file(uploaded_file) -> list:
    """Read text file and return list of lines."""
    content = uploaded_file.read()
    text_content = ""
    for encoding in ("utf-8", "latin-1"):
        try:
            text_content = content.decode(encoding)
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    
    if not text_content:
        text_content = str(content)
    
    # Split by lines and filter out empty lines
    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    return lines if lines else [text_content]

def read_csv_file(uploaded_file) -> list:
    """Extract text from CSV file."""
    df = pd.read_csv(uploaded_file)
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    
    if text_cols:
        return df[text_cols[0]].astype(str).tolist()
    
    return df.astype(str).agg(" | ".join, axis=1).tolist()

def read_json_file(uploaded_file) -> list:
    """Extract text from JSON file."""
    obj = json.loads(uploaded_file.read())
    
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and "text" in item:
                out.append(item["text"])
            else:
                out.append(json.dumps(item) if isinstance(item, dict) else str(item))
        return out
    
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list):
                return [str(x) for x in value]
        return [json.dumps(obj)]
    
    return [str(obj)]

def read_pdf_file(uploaded_file) -> list:
    """Extract text from PDF file."""
    if not PyPDF2:
        raise RuntimeError("PyPDF2 not installed. Install via: pip install PyPDF2")
    
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    return [page.extract_text() for page in pdf_reader.pages if page.extract_text()]

def create_sentiment_gauge(sentiment: str, confidence: int):
    """Create a gauge chart for sentiment visualization."""
    color = SENTIMENT_COLORS.get(sentiment, "#6b7280")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{sentiment.upper()}</b>", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#fee2e2'},
                {'range': [33, 66], 'color': '#fef3c7'},
                {'range': [66, 100], 'color': '#d1fae5'}
            ]
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_sentiment_distribution(results: list):
    """Create pie chart for sentiment distribution."""
    sentiment_counts = Counter([r.get("sentiment", "neutral") for r in results])
    
    fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_counts.keys()),
        values=list(sentiment_counts.values()),
        hole=0.4,
        marker=dict(colors=[SENTIMENT_COLORS.get(s, "#6b7280") for s in sentiment_counts.keys()]),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="📈 Sentiment Distribution",
        height=400,
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_confidence_distribution(results: list):
    """Create histogram for confidence distribution."""
    confidences = [r.get("confidence", 0) for r in results]
    sentiments = [r.get("sentiment", "neutral") for r in results]
    
    fig = go.Figure()
    
    for sentiment in set(sentiments):
        sentiment_confidences = [c for c, s in zip(confidences, sentiments) if s == sentiment]
        fig.add_trace(go.Histogram(
            x=sentiment_confidences,
            name=sentiment.capitalize(),
            marker_color=SENTIMENT_COLORS.get(sentiment, "#6b7280"),
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Count",
        barmode='overlay',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_keywords_wordcloud(results: list):
    """Create bar chart for top keywords."""
    all_keywords = []
    for result in results:
        all_keywords.extend(result.get("keywords", []))
    
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(15))
    
    if not top_keywords:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_keywords.values()),
            y=list(top_keywords.keys()),
            orientation='h',
            marker=dict(
                color=list(top_keywords.values()),
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title="🏷️ Top Keywords Extracted",
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# UI Configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Model:** {MODEL}")
    st.markdown("---")
    st.header("📤 Export Options")
    do_csv = st.checkbox("Enable CSV export", value=True)
    do_json = st.checkbox("Enable JSON export", value=True)
    st.markdown("---")
    st.caption("ℹ️ API key configured via environment")

# Get API key from environment
openai_key = os.getenv("OPENAI_API_KEY", "")

# Header
st.markdown("# 🧠 Sentiment Analysis Dashboard")
st.markdown("AI-powered text sentiment analysis with comprehensive visualizations")
st.markdown("---")

# Input area
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📝 Input")
    st.caption("Enter text manually or upload files (.txt, .csv, .json, .pdf)")
    text_input = st.text_area("Enter text for analysis", height=220, max_chars=10000, placeholder="Type or paste your text here...")
    uploaded_files = st.file_uploader("Upload files (multiple)", accept_multiple_files=True)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        analyze_single = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col_b:
        analyze_batch_btn = st.button("Run Batch Analysis", use_container_width=True)

with col_right:
    st.subheader("📊 Analysis Features")
    st.markdown("""
    - 🎯 **Sentiment Classification**: Positive, Neutral, Negative
    - 🏷️ **Keyword Extraction**: Key terms and phrases
    - 📈 **Sentiment Distribution**: Visual breakdown
    - 🔍 **Explanation**: AI-generated insights
    - 🔄 **Comparative Analysis**: Batch processing
    """)

# Initialize OpenAI client
client = None
try:
    client = get_client(openai_key or None)
except Exception as e:
    st.sidebar.error(f"🚨 {e}")

# Action handlers - MUST BE BEFORE DISPLAY
if analyze_single:
    if not client:
        st.error("❌ OpenAI API key is required (check sidebar).")
    elif not text_input or not text_input.strip():
        st.error("❌ Please provide text to analyze.")
    else:
        try:
            with st.spinner("🔄 Analyzing text..."):
                result = analyze_text(client, text_input)
            
            # Store in session state
            st.session_state.analysis_result = result
            st.session_state.analyzed_text = text_input
            st.session_state.batch_results = None  # Clear batch results
            st.success("✅ Analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Error analyzing text: {e}")
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc())

if analyze_batch_btn:
    texts = []
    
    # Extract from text area
    if text_input and text_input.strip():
        if "\n" in text_input.strip():
            texts.extend(line.strip() for line in text_input.splitlines() if line.strip())
        else:
            texts.append(text_input.strip())
    
    # Extract from files
    for file in uploaded_files or []:
        filename_lower = file.name.lower()
        try:
            if filename_lower.endswith(".txt"):
                texts.extend(read_text_file(file))
            elif filename_lower.endswith(".csv"):
                texts.extend(read_csv_file(file))
            elif filename_lower.endswith(".json"):
                texts.extend(read_json_file(file))
            elif filename_lower.endswith(".pdf"):
                texts.extend(read_pdf_file(file))
            else:
                texts.extend(read_text_file(file))
        except Exception as e:
            st.warning(f"⚠️ Failed to parse {file.name}: {e}")
    
    if not texts:
        st.error("❌ No texts found for batch processing.")
    elif not client:
        st.error("❌ OpenAI API key is required (check sidebar).")
    else:
        try:
            with st.spinner(f"🔄 Analyzing {len(texts)} items..."):
                results = analyze_batch(client, texts)
            
            # Store in session state
            st.session_state.batch_results = results
            st.session_state.analyzed_texts = texts
            st.session_state.analysis_result = None  # Clear single result
            st.success("✅ Batch analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Batch processing error: {e}")
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc())

# Display results - AFTER ACTION HANDLERS
st.markdown("---")

# Display single result
if st.session_state.analysis_result is not None:
    st.markdown("## 📊 Analysis Results")
    
    res = st.session_state.analysis_result
    text = st.session_state.analyzed_text
    
    # Top metrics
    col1, col2, col3 = st.columns(3)
    
    sentiment = res.get("sentiment", "neutral")
    confidence = res.get("confidence", 0)
    
    with col1:
        st.metric("🎯 Sentiment", sentiment.upper(), delta=None)
    with col2:
        st.metric("📊 Confidence", f"{confidence}%", delta=None)
    with col3:
        st.metric("🏷️ Keywords", len(res.get("keywords", [])), delta=None)
    
    st.markdown("---")
    
    # Visualization and details
    viz_col, detail_col = st.columns([1, 1])
    
    with viz_col:
        st.markdown("### 🎯 Sentiment Classification")
        fig = create_sentiment_gauge(sentiment, confidence)
        st.plotly_chart(fig, use_container_width=True)
    
    with detail_col:
        st.markdown("### 🔍 Explanation")
        st.info(res.get("explanation", "No explanation provided."))
        
        st.markdown("### 🏷️ Keyword Extraction")
        keywords = res.get("keywords", [])
        if keywords:
            keyword_html = " ".join([
                f"<span style='display:inline-block;background:{SENTIMENT_COLORS.get(sentiment, '#6b7280')}20;color:{SENTIMENT_COLORS.get(sentiment, '#6b7280')};padding:6px 12px;border-radius:20px;margin:4px;font-size:14px'>{kw}</span>"
                for kw in keywords
            ])
            st.markdown(keyword_html, unsafe_allow_html=True)
        else:
            st.write("No keywords extracted.")
    
    st.markdown("---")
    
    # Export section
    st.markdown("### ⬇ Export Results")
    df_row = pd.DataFrame([{
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "keywords": ", ".join(keywords),
        "explanation": res.get("explanation")
    }])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if do_csv:
            st.download_button("📥 Export CSV", data=df_row.to_csv(index=False), file_name="sentiment_result.csv", use_container_width=True)
    with col2:
        if do_json:
            st.download_button("📥 Export JSON", data=json.dumps(res, indent=2), file_name="sentiment_result.json", use_container_width=True)

# Display batch results
elif st.session_state.batch_results is not None:
    st.markdown("## 📊 Batch Analysis Results")
    
    results = st.session_state.batch_results
    texts = st.session_state.analyzed_texts
    
    # Top metrics
    total = len(results)
    sentiments = [r.get("sentiment", "neutral") for r in results]
    avg_confidence = sum([r.get("confidence", 0) for r in results]) / total if total > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Analyzed", total)
    with col2:
        st.metric("😊 Positive", sentiments.count("positive"))
    with col3:
        st.metric("😐 Neutral", sentiments.count("neutral"))
    with col4:
        st.metric("😞 Negative", sentiments.count("negative"))
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 🔄 Comparative Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = create_sentiment_distribution(results)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_conf = create_confidence_distribution(results)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Keywords analysis
    st.markdown("---")
    fig_keywords = create_keywords_wordcloud(results)
    if fig_keywords:
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed table
    st.markdown("## 📋 Detailed Results")
    rows = [
        {
            "text_preview": (text[:TEXT_PREVIEW_LEN] + "...") if len(text) > TEXT_PREVIEW_LEN else text,
            "sentiment": result.get("sentiment"),
            "confidence": result.get("confidence"),
            "keywords": ", ".join(result.get("keywords", [])),
            "explanation": result.get("explanation")
        }
        for text, result in zip(texts, results)
    ]
    
    df = pd.DataFrame(rows)
    
    # Color code by sentiment
    def highlight_sentiment(row):
        color = SENTIMENT_COLORS.get(row['sentiment'], '#6b7280')
        return [f'background-color: {color}20' for _ in row]
    
    st.dataframe(df.style.apply(highlight_sentiment, axis=1), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Export options
    st.markdown("### ⬇ Export Results")
    col1, col2 = st.columns([1, 1])
    with col1:
        if do_csv:
            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="batch_results.csv", use_container_width=True)
    with col2:
        if do_json:
            st.download_button("📥 Download JSON", json.dumps(results, indent=2), file_name="batch_results.json", use_container_width=True)

else:
    # Show placeholder when no results
    st.info("👆 Enter text and click 'Analyze Sentiment' or 'Run Batch Analysis' to see results here.")