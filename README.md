# 🧠 Sentiment Analysis Dashboard

A lightweight **AI-powered sentiment analysis** app built with **Python** and **Streamlit**.

This dashboard lets you:
- Analyze the sentiment of text 🎯  
- Extract keywords 🏷️  
- View confidence scores 📊  
- Run batch text analysis  
- Upload files for automatic processing  

---

## 🚀 Features

✅ Single text sentiment analysis  
✅ Batch processing for multiple inputs  
✅ File uploads: `.txt`, `.csv`, `.json`, `.pdf`  
✅ Sentiment gauge visualization  
✅ Confidence score charts  
✅ Keyword frequency bar chart  
✅ Results export (CSV / JSON)

---

## 🧠 Technologies Used

| Component | Library |
|----------|---------|
| Web UI | Streamlit |
| AI Model | OpenAI API |
| Charts | Plotly |
| File Parsing | Pandas, PyPDF2 |
| Environment Vars | python-dotenv |

---

## 🖥️ Local Setup

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Add your API key

Create a .env file:

OPENAI_API_KEY=your_api_key_here

3️⃣ Run the app
streamlit run app.py

📂 Project Structure
app.py         # Main application
README.md      # Documentation
.env           # OpenAI key (optional)

📤 How to Use

Enter text or upload files

Click Analyze Sentiment or Run Batch Analysis

View charts, keywords, and explanations

Download CSV/JSON results if needed
