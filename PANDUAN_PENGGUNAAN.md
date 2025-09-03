# 🚀 Panduan Penggunaan BlockAGI dengan GROQ API

## 📋 Daftar Isi
1. [Persiapan Awal](#persiapan-awal)
2. [Setup GROQ API](#setup-groq-api)
3. [Instalasi Project](#instalasi-project)
4. [Konfigurasi Environment](#konfigurasi-environment)
5. [Menjalankan BlockAGI](#menjalankan-blockagi)
6. [Menggunakan Web Interface](#menggunakan-web-interface)
7. [Troubleshooting](#troubleshooting)
8. [Tips dan Trik](#tips-dan-trik)

## 🎯 Persiapan Awal

### Software yang Diperlukan
- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **Poetry** - Dependency manager untuk Python
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/)

### Hardware yang Direkomendasikan
- **RAM**: Minimal 4GB, direkomendasikan 8GB+
- **Storage**: 2GB+ free space
- **Internet**: Koneksi stabil untuk API calls

## 🔑 Setup GROQ API

### 1. Daftar Akun GROQ
1. Kunjungi [console.groq.com](https://console.groq.com)
2. Klik "Sign Up" dan buat akun baru
3. Verifikasi email Anda

### 2. Dapatkan API Key
1. Login ke [console.groq.com](https://console.groq.com)
2. Klik "API Keys" di sidebar kiri
3. Klik "Create API Key"
4. Beri nama untuk API key (misal: "BlockAGI")
5. Copy API key yang dihasilkan

### 3. Pilih Model GROQ
GROQ menyediakan beberapa model yang dapat digunakan:

| Model | Context Window | Keunggulan | Penggunaan |
|-------|----------------|------------|------------|
| `llama3-8b-8192` | 8K tokens | Cepat, efisien | Research umum |
| `llama3-70b-8192` | 8K tokens | Lebih akurat | Research kompleks |
| `mixtral-8x7b-32768` | 32K tokens | Context panjang | Research mendalam |
| `gemma2-9b-it` | 8K tokens | Multilingual | Research internasional |

**Rekomendasi**: Mulai dengan `llama3-8b-8192` untuk testing, lalu upgrade ke model yang lebih powerful sesuai kebutuhan.

## 🛠️ Instalasi Project

### 1. Clone Repository
```bash
# Clone project dari GitHub
git clone https://github.com/blockpipe/BlockAGI.git

# Masuk ke directory project
cd BlockAGI
```

### 2. Install Python Dependencies
```bash
# Install Poetry (jika belum ada)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies Python
poetry install

# Install Playwright dependencies
poetry run playwright install
```

### 3. Install Frontend Dependencies
```bash
# Masuk ke directory UI
cd ui

# Install dependencies
npm install
# atau
yarn install

# Build frontend
npm run build
# atau
yarn build

# Kembali ke root directory
cd ..
```

## ⚙️ Konfigurasi Environment

### 1. Buat File Environment
```bash
# Copy file environment example
cp env.example .env

# Edit file environment
nano .env
# atau
code .env
```

### 2. Isi Konfigurasi GROQ
```env
# GROQ API Configuration
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=llama3-8b-8192

# Web Server Configuration
WEB_HOST=localhost
WEB_PORT=8000

# BlockAGI Configuration
BLOCKAGI_AGENT_ROLE=Research Assistant
BLOCKAGI_ITERATION_COUNT=3

# Research Objectives
BLOCKAGI_OBJECTIVE_1=Bitcoin adoption trends
BLOCKAGI_OBJECTIVE_2=DeFi security analysis
BLOCKAGI_OBJECTIVE_3=Blockchain scalability solutions
```

### 3. Verifikasi Konfigurasi
```bash
# Test GROQ API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     "https://api.groq.com/openai/v1/models"

# Output yang diharapkan: list model yang tersedia
```

## 🚀 Menjalankan BlockAGI

### 1. Jalankan Backend
```bash
# Dari root directory
poetry run python main.py
```

**Output yang diharapkan:**
```
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Started reloader process [12346]
INFO:     Started browser process [12347]
```

### 2. Akses Web Interface
- Browser akan terbuka otomatis ke `http://localhost:8000`
- Jika tidak, buka manual: `http://localhost:8000`

### 3. Monitor Progress
- **Objectives Tab**: Lihat tujuan penelitian
- **Agent Logs**: Monitor aktivitas agent real-time
- **Findings**: Hasil penelitian yang ditemukan
- **Narratives**: Laporan lengkap dalam markdown

## 🌐 Menggunakan Web Interface

### 1. **Objectives Tab** 🎯
- **User-Defined Objectives**: Tujuan penelitian yang Anda set
- **Generated Objectives**: Sub-tujuan yang dibuat agent
- **Expertise Level**: Progress pemahaman agent (0-100%)

### 2. **Agent Logs** 📝
- **Real-time Updates**: Lihat aktivitas agent secara langsung
- **Tool Usage**: Monitor tools yang digunakan
- **Decision Process**: Pahami logika agent

### 3. **Findings** 🔍
- **Research Results**: Hasil penelitian yang ditemukan
- **Citations**: Sumber informasi yang digunakan
- **Generated Objectives**: Sub-tujuan yang dibuat

### 4. **Narratives** 📚
- **Comprehensive Reports**: Laporan lengkap dalam markdown
- **Structured Content**: Informasi terorganisir dengan baik
- **Export Ready**: Siap untuk di-export ke format lain

## 🔧 Troubleshooting

### 1. **GROQ API Key Error**
```bash
# Error: Invalid API key
# Solusi: Pastikan API key benar dan aktif

# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     "https://api.groq.com/openai/v1/models"

# Jika error 401: API key salah
# Jika error 403: API key tidak aktif
# Jika error 429: Rate limit exceeded
```

### 2. **Port Already in Use**
```bash
# Error: Address already in use
# Solusi: Ganti port atau kill process

# Check port usage
netstat -tulpn | grep :8000

# Kill process
kill -9 <PID>

# Atau ganti port di .env
WEB_PORT=8001
```

### 3. **Dependency Issues**
```bash
# Error: Module not found
# Solusi: Reinstall dependencies

# Clear Poetry cache
poetry cache clear . --all

# Reinstall
poetry install --sync
```

### 4. **Playwright Issues**
```bash
# Error: Browser not found
# Solusi: Reinstall Playwright

poetry run playwright install --force
```

### 5. **Memory Issues**
```bash
# Error: Out of memory
# Solusi: Kurangi iteration count

# Di .env
BLOCKAGI_ITERATION_COUNT=2  # Kurangi dari 3 ke 2
```

## 💡 Tips dan Trik

### 1. **Optimasi Performance**
- **Iteration Count**: Mulai dengan 2-3, tingkatkan sesuai kebutuhan
- **Model Selection**: Gunakan model yang sesuai dengan kompleksitas research
- **Resource Management**: Monitor memory usage

### 2. **Research Objectives yang Efektif**
```env
# Contoh objectives yang baik
BLOCKAGI_OBJECTIVE_1=Bitcoin adoption in Southeast Asia
BLOCKAGI_OBJECTIVE_2=DeFi security vulnerabilities 2024
BLOCKAGI_OBJECTIVE_3=Layer 2 scaling solutions comparison

# Contoh objectives yang kurang efektif
BLOCKAGI_OBJECTIVE_1=Blockchain  # Terlalu umum
BLOCKAGI_OBJECTIVE_2=Crypto      # Terlalu singkat
```

### 3. **Custom Agent Roles**
```env
# Untuk research akademik
BLOCKAGI_AGENT_ROLE=Academic Research Assistant

# Untuk business intelligence
BLOCKAGI_AGENT_ROLE=Business Intelligence Analyst

# Untuk technical research
BLOCKAGI_AGENT_ROLE=Technical Research Specialist
```

### 4. **Monitoring dan Logging**
```bash
# Lihat logs real-time
tail -f logs/blockagi.log

# Monitor resource usage
htop
# atau
top
```

### 5. **Backup dan Recovery**
```bash
# Backup environment
cp .env .env.backup

# Backup research results
cp -r research_results/ research_results_backup/

# Restore jika diperlukan
cp .env.backup .env
```

## 📊 Monitoring Progress

### 1. **Real-time Metrics**
- **Progress Bar**: Lihat completion percentage
- **Time Elapsed**: Waktu yang sudah berjalan
- **Current Step**: Tahap yang sedang berjalan

### 2. **Performance Indicators**
- **API Response Time**: Kecepatan GROQ API
- **Tool Execution**: Waktu eksekusi tools
- **Memory Usage**: Penggunaan memory

### 3. **Quality Metrics**
- **Expertise Level**: Tingkat pemahaman agent
- **Citation Count**: Jumlah sumber yang digunakan
- **Content Length**: Panjang laporan yang dihasilkan

## 🔄 Advanced Usage

### 1. **Custom Tools**
```python
# Tambahkan tool kustom di blockagi/tools/
from langchain.tools import BaseTool

class CustomTool(BaseTool):
    name = "Custom Tool"
    description = "Description of your custom tool"
    
    def _run(self, query: str):
        # Implementasi tool
        return "Result"
```

### 2. **Batch Processing**
```bash
# Jalankan multiple research sessions
for topic in "bitcoin" "ethereum" "defi"; do
    BLOCKAGI_OBJECTIVE_1="$topic analysis" poetry run python main.py
done
```

### 3. **Scheduled Research**
```bash
# Gunakan cron untuk research otomatis
# Edit crontab
crontab -e

# Tambahkan job (research setiap hari jam 9 pagi)
0 9 * * * cd /path/to/BlockAGI && poetry run python main.py
```

## 🆘 Getting Help

### 1. **Documentation**
- [BlockAGI GitHub](https://github.com/blockpipe/BlockAGI)
- [GROQ API Docs](https://console.groq.com/docs)
- [LangChain Docs](https://python.langchain.com/)

### 2. **Community Support**
- [GitHub Issues](https://github.com/blockpipe/BlockAGI/issues)
- [Discord Community](https://discord.gg/K3TWumAtZV)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/blockagi)

### 3. **Debug Mode**
```bash
# Enable debug logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
poetry run python -m pdb main.py
```

---

## 🎉 Selamat! Anda telah berhasil setup BlockAGI dengan GROQ API

**BlockAGI** sekarang siap digunakan untuk penelitian otomatis dengan AI yang powerful dan efisien. Jangan ragu untuk bereksperimen dengan berbagai objectives dan agent roles untuk mendapatkan hasil terbaik!

**Happy Researching! 🚀**
