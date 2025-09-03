# 📚 Dokumentasi Lengkap BlockAGI

## 🎯 Overview

**BlockAGI** adalah AI Research Agent open-source yang dibangun dengan Python dan LangChain. Project ini dirancang untuk melakukan penelitian otomatis dengan domain yang dapat dikustomisasi, terutama fokus pada cryptocurrency namun dapat digunakan untuk berbagai bidang penelitian lainnya.

## 🏗️ Arsitektur Sistem

### Komponen Utama

#### 1. **Python Engine (Backend)**
- **Core**: Berbasis LangChain dengan modifikasi custom
- **Memory**: Resource Pool sebagai short-term memory
- **Tools**: Search engines, web scraping, dan tools kustom
- **Chains**: Sistem PRUNE (Plan, Research, Update, Narrate, Evaluate)

#### 2. **Web UI (Frontend)**
- **Framework**: Next.js dengan TypeScript
- **Styling**: Tailwind CSS
- **Real-time**: Monitoring progress agent secara langsung
- **Interactive**: Interface yang user-friendly

#### 3. **AI Agent Core**
- **LLM Integration**: OpenAI API (dapat diganti dengan GROQ)
- **Autonomous Decision**: Agent dapat merencanakan dan mengeksekusi penelitian
- **Iterative Learning**: Self-improvement melalui multiple iterations

### Arsitektur Data Flow

```
User Input → Objectives → Plan Chain → Research Chain → 
Update Resources → Narrate Chain → Evaluate Chain → 
Next Iteration (jika ada)
```

## 🔧 Setup dan Instalasi

### Prasyarat Sistem

#### Software Requirements
```bash
# Python 3.9+
python --version

# Poetry (dependency management)
curl -sSL https://install.python-poetry.org | python3 -

# Node.js 18+ dan npm
node --version
npm --version

# Git
git --version
```

#### Hardware Requirements
- **RAM**: Minimal 4GB, direkomendasikan 8GB+
- **Storage**: 2GB+ free space
- **Internet**: Koneksi stabil untuk API calls

### Langkah Instalasi

#### 1. Clone Repository
```bash
git clone https://github.com/blockpipe/BlockAGI.git
cd BlockAGI
```

#### 2. Setup Backend (Python)
```bash
# Install dependencies dengan Poetry
poetry install

# Install Playwright dependencies
poetry run playwright install
```

#### 3. Setup Frontend (Next.js)
```bash
cd ui
npm install
# atau
yarn install
```

#### 4. Build Frontend
```bash
npm run build
# atau
yarn build
```

#### 5. Setup Environment Variables
```bash
# Copy file environment
cp .env.example .env

# Edit file .env dengan API keys
nano .env
```

### Konfigurasi Environment Variables

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Web Server
WEB_HOST=localhost
WEB_PORT=8000

# BlockAGI Configuration
BLOCKAGI_AGENT_ROLE=Research Assistant
BLOCKAGI_ITERATION_COUNT=3

# Research Objectives
BLOCKAGI_OBJECTIVE_1=Bitcoin adoption trends
BLOCKAGI_OBJECTIVE_2=DeFi security analysis

# Optional: Google Search API
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

## 🚀 Cara Menjalankan

### 1. Menjalankan Backend
```bash
# Dari root directory
poetry run python main.py

# Atau dengan parameter custom
poetry run python main.py --host 0.0.0.0 --port 8080
```

### 2. Mengakses Web Interface
- Buka browser dan akses: `http://localhost:8000`
- Interface akan terbuka otomatis setelah backend berjalan

### 3. Monitoring Progress
- **Objectives Tab**: Lihat tujuan penelitian dan expertise level
- **Agent Logs**: Monitor aktivitas agent secara real-time
- **Findings**: Hasil penelitian yang ditemukan
- **Narratives**: Laporan lengkap dalam format markdown

## 🛠️ Integrasi GROQ API

### Mengganti OpenAI dengan GROQ

#### 1. Update Dependencies
```toml
# pyproject.toml
[tool.poetry.dependencies]
groq = "^0.4.0"  # Tambahkan dependency GROQ
```

#### 2. Update Code untuk GROQ
```python
# blockagi/run.py
from groq import Groq

def run_blockagi(
    agent_role,
    groq_api_key,  # Ganti openai_api_key
    groq_model,    # Ganti openai_model
    resource_pool,
    objectives,
    blockagi_callback,
    llm_callback,
    iteration_count,
):
    # ... existing code ...
    
    # Ganti ChatOpenAI dengan Groq
    llm = Groq(
        api_key=groq_api_key,
        model=groq_model,  # misal: "llama3-8b-8192"
        temperature=0.8,
        streaming=True,
        callbacks=[llm_callback],
    )
```

#### 3. Update Environment Variables
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

#### 4. Update Main.py
```python
# main.py
def main(
    # ... existing parameters ...
    groq_api_key: str = typer.Option(envvar="GROQ_API_KEY"),
    groq_model: str = typer.Option(envvar="GROQ_MODEL"),
):
    # ... existing code ...
    app.state.groq_api_key = groq_api_key
    app.state.groq_model = groq_model
```

## 📊 Fitur dan Kemampuan

### Core Features
- **🔄 Automated Research**: Penelitian otomatis berdasarkan objectives
- **🔍 Multi-source Search**: DuckDuckGo, Google, dan web scraping
- **📝 Narrative Generation**: Laporan otomatis dalam format markdown
- **🎯 Self-evaluation**: Agent dapat mengevaluasi dan memperbaiki diri
- **🌐 Real-time Web UI**: Monitoring progress secara langsung

### Tools yang Tersedia
1. **DuckDuckGo Search**: Pencarian informasi real-time
2. **Web Scraping**: Ekstraksi konten dari website
3. **Google Search**: Pencarian tambahan (opsional)
4. **Resource Pool**: Manajemen sumber informasi

### Customization Options
- **Agent Role**: Dapat dikustomisasi sesuai kebutuhan
- **Research Domain**: Fleksibel untuk berbagai bidang
- **Iteration Count**: Jumlah iterasi penelitian
- **Tools**: Dapat menambah tools kustom

## 🔍 Use Cases

### 1. **Cryptocurrency Research**
- Market analysis
- Technology trends
- Security assessment
- Regulatory compliance

### 2. **Academic Research**
- Literature review
- Data gathering
- Trend analysis
- Comparative studies

### 3. **Business Intelligence**
- Market research
- Competitor analysis
- Industry trends
- Risk assessment

### 4. **Technical Research**
- Technology evaluation
- Security research
- Performance analysis
- Best practices

## 🚨 Troubleshooting

### Common Issues

#### 1. **API Key Errors**
```bash
# Pastikan API key sudah benar
echo $GROQ_API_KEY

# Test API connection
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     "https://api.groq.com/openai/v1/models"
```

#### 2. **Dependency Issues**
```bash
# Clear Poetry cache
poetry cache clear . --all

# Reinstall dependencies
poetry install --sync
```

#### 3. **Port Already in Use**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Kill process using port
kill -9 <PID>
```

#### 4. **Playwright Issues**
```bash
# Reinstall Playwright
poetry run playwright install --force
```

### Performance Optimization

#### 1. **Memory Management**
- Monitor resource pool size
- Implement cleanup mechanisms
- Optimize web scraping limits

#### 2. **API Rate Limiting**
- Implement exponential backoff
- Add request queuing
- Monitor API usage

#### 3. **Caching Strategy**
- Cache search results
- Implement resource pooling
- Store intermediate results

## 🔒 Security Considerations

### Best Practices
1. **API Key Security**
   - Jangan commit API keys ke repository
   - Gunakan environment variables
   - Rotate keys secara berkala

2. **Web Scraping Ethics**
   - Respect robots.txt
   - Implement rate limiting
   - Handle errors gracefully

3. **Data Privacy**
   - Tidak menyimpan data sensitif
   - Implement data retention policies
   - Secure communication channels

## 📈 Monitoring dan Logging

### Log Types
1. **Agent Logs**: Aktivitas agent
2. **LLM Logs**: Input/output LLM
3. **Tool Logs**: Eksekusi tools
4. **Error Logs**: Error handling

### Metrics to Monitor
- Response time
- Success rate
- Resource usage
- API costs

## 🚀 Pengembangan Lanjutan

### Fitur yang Direkomendasikan

#### 1. **Enhanced Memory Management**
- Vector database integration
- Long-term memory storage
- Context window optimization

#### 2. **Advanced Tools**
- Database connectors
- API integrations
- File processing tools
- Image analysis

#### 3. **Collaboration Features**
- Multi-user support
- Project sharing
- Version control
- Team collaboration

#### 4. **Analytics Dashboard**
- Research metrics
- Performance analytics
- Cost tracking
- Usage patterns

#### 5. **Export Capabilities**
- PDF generation
- Word document export
- Presentation slides
- Data visualization

## 🧪 Testing dan Quality Assurance

### Testing Strategy
1. **Unit Tests**: Individual components
2. **Integration Tests**: Chain workflows
3. **End-to-End Tests**: Complete workflows
4. **Performance Tests**: Load testing

### Code Quality
- **Linting**: Black, flake8
- **Type Checking**: MyPy
- **Documentation**: Sphinx
- **Coverage**: pytest-cov

## 📚 Referensi dan Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [GROQ API Documentation](https://console.groq.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)

### Community
- [GitHub Repository](https://github.com/blockpipe/BlockAGI)
- [Discord Community](https://discord.gg/K3TWumAtZV)
- [Issues & Discussions](https://github.com/blockpipe/BlockAGI/issues)

### Related Projects
- [BabyAGI](https://github.com/yoheinakajima/babyagi)
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
- [LangChain](https://github.com/hwchase17/langchain)

## 📄 License

BlockAGI dilisensikan di bawah Apache License 2.0. Lihat file [LICENSE](LICENSE) untuk detail lengkap.

## 🤝 Contributing

Kontribusi sangat dihargai! Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan lengkap.

---

**Dibuat dengan ❤️ oleh tim BlockAGI**
