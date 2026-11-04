# 🧠 RAG — Retrieval-Augmented Generation from Scratch

A **simple, framework-free** RAG system built with Python and [Ollama](https://ollama.com). No LangChain, no vector databases — just `requests`, `numpy`, and clear code you can actually understand.

---

## 🤔 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that makes a Large Language Model (LLM) answer questions using **your own documents** instead of its general training data.

Here's the idea in plain English:

1. **You give it your documents** — text files, handbooks, notes, anything.
2. **It splits them into chunks** — small, meaningful paragraphs.
3. **It converts each chunk into a vector** (a list of numbers that captures the meaning).
4. **When you ask a question**, it converts your question into a vector too.
5. **It finds the most relevant chunks** by comparing vectors (cosine similarity).
6. **It sends those chunks + your question to the LLM**, which answers based only on the provided context.

> 💡 Think of it like giving the AI a cheat sheet before an exam — it can only use what's on the sheet.

---

## 📁 Project Structure

```
RAG/
├── rag.py              # The entire RAG system (~60 lines)
├── docs/               # Your knowledge base (Markdown files)
│   └── *.md            # Add your documents here
├── pyproject.toml      # Project config & dependencies
├── .gitignore          # Files excluded from git
└── README.md           # You are here
```

---

## 🚀 How It Works — Step by Step

### Step 1: Load & Chunk Documents

```python
chunks = []
for file in Path('docs').glob('*.md'):
    chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]
```

- Reads every `.md` file from the `docs/` folder.
- Splits each file by **double newlines** (`\n\n`) into paragraphs.
- Each paragraph becomes a **chunk**.

### Step 2: Embed the Chunks

```python
chunk_vectors = embed(chunks)
```

- Sends all chunks to Ollama's embedding model (`nomic-embed-text`).
- Each chunk becomes a **vector** — a numerical representation of its meaning.
- Similar meanings → similar vectors.

### Step 3: Retrieve Relevant Chunks

```python
def retrieve(question, k=3):
    q = embed([question])[0]
    scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]
```

- Embeds your question into a vector.
- Calculates **cosine similarity** between your question and every chunk.
- Returns the top `k` (default: 3) most relevant chunks.

### Step 4: Ask the LLM with Context

```python
prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}"""

print(ask_llm(prompt))
```

- Builds a prompt with the retrieved context + your question.
- The LLM answers **only** from the context — no hallucination.

---

## ⚙️ Setup & Installation

### Prerequisites

- **Python 3.14+**
- **[Ollama](https://ollama.com)** installed and running
- **[uv](https://docs.astral.sh/uv/)** (recommended Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/Iyedchebbi/RAG.git
cd RAG
```

### 2. Install Dependencies

Using **uv** (recommended):

```bash
uv sync
```

Or using **pip**:

```bash
pip install requests numpy
```

### 3. Pull the Required Ollama Models

```bash
# Embedding model (converts text → vectors)
ollama pull nomic-embed-text

# Chat model (answers your questions)
ollama pull qwen3:4b-instruct
```

### 4. Add Your Documents

Place your `.md` (Markdown) files in the `docs/` folder. The system reads all `.md` files from this directory.

### 5. Run It

```bash
python rag.py
```

You'll see:

```
Indexed 42 chunks

Ask: 
```

Type your question and get an answer based on your documents!

---

## 🔧 Configuration

You can change the models at the top of `rag.py`:

```python
OLLAMA = "http://localhost:11434"    # Ollama server address
EMBED_MODEL = "nomic-embed-text"     # Embedding model
CHAT_MODEL = "qwen3:4b-instruct"    # Chat/answer model
```

| Setting | What it does | Alternatives |
|---|---|---|
| `EMBED_MODEL` | Converts text to vectors | `mxbai-embed-large`, `all-minilm` |
| `CHAT_MODEL` | Generates answers | `llama3`, `mistral`, `gemma2` |
| `k=3` in `retrieve()` | Number of chunks to retrieve | Increase for more context |

---

## 🧩 Key Concepts Explained

| Concept | Simple Explanation |
|---|---|
| **Embedding** | Converting text into numbers (vectors) that capture meaning |
| **Vector** | A list of numbers representing a piece of text |
| **Cosine Similarity** | A way to measure how similar two vectors are (1 = identical, 0 = unrelated) |
| **Chunk** | A small piece of your document (usually a paragraph) |
| **Context Window** | The text you give the LLM along with your question |
| **Retrieval** | Finding the most relevant chunks for a given question |
| **Grounding** | Making the LLM answer only from provided context, not its training data |

---

## 📝 Example

```
Ask: What services does the business offer?

>> The business offers website design and development, advertising
   creative production, AI-powered UGC-style content, and video editing.
```

---

## 🏗️ Why Build RAG from Scratch?

- **Learn the fundamentals** — understand what frameworks like LangChain do under the hood.
- **No magic, no abstraction** — every line of code is visible and understandable.
- **Lightweight** — only 2 dependencies (`requests` and `numpy`).
- **Customizable** — easy to modify, extend, or integrate into your own projects.

---

## 📄 License

This project is open source. Feel free to use, modify, and learn from it.

---

**Built by [Iyed Chebbi](https://github.com/Iyedchebbi)** 🇹🇳
