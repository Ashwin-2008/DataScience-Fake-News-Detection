import streamlit as st
import requests
import cohere
from sentence_transformers import SentenceTransformer, util
import re


API_KEY
SEARCH_ENGINE_ID
COHERE_API_KEY


co = cohere.Client(COHERE_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")


CREDIBLE_SOURCES = [
    "bbc", "cnn", "reuters", "associated press", "ap news",
    "the hindu", "times of india", "hindustan times", "ndtv",
    "al jazeera", "bloomberg", "wall street journal", "forbes",
    "economic times", "india today", "mint", "business standard",
    "the times", "the guardian", "washington post", "new york times"
]

HIGH_CRED_SOURCES = ["bbc", "reuters", "associated press", "ap news"]

SUSPICIOUS_KEYWORDS = ["blogspot", "wordpress", "medium.com", "quora", "reddit"]

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(text.split())

def get_domain_credibility_boost(url, text):
    boost = 0.0
    domain_lower = url.lower()
    text_lower = text.lower()

    for source in HIGH_CRED_SOURCES:
        if source in domain_lower or source in text_lower:
            boost += 0.35
            break
    for source in CREDIBLE_SOURCES:
        if source in domain_lower or source in text_lower:
            boost += 0.15
            break
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in domain_lower:
            boost -= 0.2
            break
    return min(0.5, boost)

def calculate_semantic_similarity(news_input, articles_data):
    news_clean = clean_text(news_input)
    news_vec = model.encode(news_clean, convert_to_tensor=True)
    scores = []

    for snippet, title, link in articles_data:
        title_clean = clean_text(title)
        snippet_clean = clean_text(snippet)
        title_vec = model.encode(title_clean, convert_to_tensor=True)
        snippet_vec = model.encode(snippet_clean, convert_to_tensor=True)

        base_sim = max(util.cos_sim(news_vec, title_vec).item(),
                       util.cos_sim(news_vec, snippet_vec).item())

        credibility_boost = get_domain_credibility_boost(link, f"{title} {snippet}")
        final_score = min(1.0, base_sim + credibility_boost)
        scores.append(final_score)

    return scores

def analyze_with_cohere(news_input, top_articles):
    try:
        context = "\n".join([f"- {art}" for art in top_articles[:3]])
        prompt = f"""
        News to verify: "{news_input}"

        Related articles from search:
        {context}

        Analyze if the news appears to be real or fake based on:
        1. Consistency with reliable sources
        2. Presence of sensational language
        3. Logical coherence
        4. Supporting evidence

        Provide analysis in 2-3 sentences.
        """
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )
        return response.generations[0].text.strip()
    except:
        return "AI analysis unavailable."

st.title(" Fake News Detection — Advanced Real-Time Verification")
st.write("AI-based real-time verification using semantic meaning + live credibility check.")

news_input = st.text_area("Enter news headline or paragraph")

if st.button("🔍 Analyze News"):
    if not news_input.strip():
        st.warning("Please enter a news headline or paragraph.")
    else:
        with st.spinner("Analyzing live data..."):
            try:
                search_url = f"https://www.googleapis.com/customsearch/v1?q={news_input}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num=8"
                resp = requests.get(search_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if "items" not in data:
                    st.error("No results found or API quota exhausted.")
                    st.stop()

                articles_data = [(item.get("snippet", ""), item.get("title", ""), item.get("link", "")) 
                                 for item in data["items"][:8]]

                scores = calculate_semantic_similarity(news_input, articles_data)

                weighted_scores = []
                for (snippet, title, link), score in zip(articles_data, scores):
                    if any(x in link.lower() for x in HIGH_CRED_SOURCES):
                        weighted_scores.append(score * 1.3)
                    else:
                        weighted_scores.append(score)
                conf = min(1.0, sum(weighted_scores)/len(weighted_scores))


                top_titles = [title for _, title, _ in articles_data[:3]]
                cohere_text = analyze_with_cohere(news_input, top_titles)
                if any(x in cohere_text.lower() for x in ["real", "credible", "verified"]):
                    conf += 0.1
                elif any(x in cohere_text.lower() for x in ["fake", "hoax", "unverified"]):
                    conf -= 0.1
                conf = min(1.0, max(0.0, conf))


                top_avg = sum(sorted(scores, reverse=True)[:3])/3
                if top_avg >= 0.6:
                    result = "REAL — Verified from credible live sources."
                    color = "green"
                elif top_avg >= 0.4:
                    result = " UNCERTAIN — Needs further verification."
                    color = "orange"
                else:
                    result = " FAKE — No credible confirmation found."
                    color = "red"

                st.markdown(f"### **Result:** <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence Score:** {conf:.2f}")

                st.markdown("###  **Top Matching Articles:**")
                for i, ((snippet, title, link), score) in enumerate(zip(articles_data[:5], scores[:5])):
                    st.markdown(f"**{i+1}. [{title}]({link})** (Score: {score:.2f})")
                    st.caption(f"{snippet[:150]}...")

                with st.expander("🤖 AI Analysis"):
                    st.write(cohere_text)

            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        st.markdown("---")
        st.caption("© 2025 Fake News Detection | Built with Semantic AI + Live Web Verification")
