# Virtual Try-On

A web-based virtual try-on application powered by AI. Upload a photo and a garment to see how it looks on you.

## 📋 Features

- **Multiple AI Models**: Choose from CatVTON, Kling Kolors, Nano Banana Pro, Qwen Image Max, or FASHN
- **Easy Upload**: Simple drag-and-drop interface for person photos and garments
- **Real-time Processing**: Get results in seconds
- **Customizable Parameters**: Adjust steps, seed, and clothing category
- **Download Results**: Save your try-on images locally

## 🚀 Quick Start

### Prerequisites

- Python 3.6+ (for local server)
- Modern web browser
- API keys from:
  - [fal.ai](https://fal.ai/dashboard/keys) (for CatVTON, Kling, Nano Banana Pro, Qwen)
  - [RunPod](https://www.runpod.io/console/api-keys) (optional, for FASHN model)

### Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/VTON.git
   cd VTON
   ```

2. **Configure API Keys**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:

   ```
   FAL_KEY=your_key_id:your_key_secret
   RP_EP=your_endpoint_id
   RP_KEY=your_runpod_key
   ```

3. **Start local server**

   ```bash
   python3 server.py
   ```
   
   This reads your `.env` file and provides the API keys to the application.

4. **Open in browser**
   ```
   http://localhost:8080/public/
   ```

## 🔧 Configuration

### API Keys

- **FAL.AI Key**: Get from [fal.ai dashboard](https://fal.ai/dashboard/keys)
  - Format: `key_id:key_secret`
  - Used for: CatVTON, Kling Kolors, Nano Banana Pro, Qwen Image Max

- **RunPod Credentials**: Required only for FASHN model
  - Endpoint ID: Your RunPod endpoint
  - API Key: From [RunPod API Keys](https://www.runpod.io/console/api-keys)

### Model Options

| Model             | API    | Cost     | Quality   |
| ----------------- | ------ | -------- | --------- |
| CatVTON           | fal.ai | ~$0.01   | Good      |
| Kling Kolors v1.5 | fal.ai | $0.07    | Excellent |
| Nano Banana Pro   | fal.ai | $0.15    | Good      |
| Qwen Image Max    | fal.ai | $0.075   | Good      |
| FASHN v1.5        | RunPod | Your GPU | Excellent |

## 📖 Usage

1. Upload a **Person** photo (clear, full-body view)
2. Upload a **Garment** image (flat-lay or on model)
3. Select your preferred **Model** and **Category**
4. Adjust **Steps** and **Seed** if desired
5. Click **Try On**
6. Download the result

## ⚙️ Advanced Configuration

### Adjust Processing Parameters

- **Steps**: 10-50 (higher = more refined, slower)
- **Seed**: 0-999999 (for reproducible results)
- **Category**: Tops, Bottoms, Dresses

## 🔐 Security

⚠️ **IMPORTANT**: Never commit `.env` file with real API keys to GitHub.

- `.env` is in `.gitignore` to prevent accidental commits
- Always use `.env.example` as a template
- Keep your API keys private and never share them

## 🐛 Troubleshooting

### CORS Errors / "Set Key" Message

- Make sure you're running `python3 server.py`, NOT `python3 -m http.server`
- The custom server.py reads your `.env` file and provides keys to the app
- Ensure your `.env` file exists in the project root with correct keys

### API Key Invalid

- Check `.env` file has correct format: `key_id:key_secret` (with colon)
- Verify keys from respective dashboards (fal.ai, RunPod)
- Restart the server after changing `.env`

### Server Not Starting

- Make sure you have Python 3.6+
- Try: `python3 --version`
- Check that the `.env` file exists
- Look at console output for errors

### Long Processing Time

- Check API status pages (normal: 30-120 seconds)
- Consider using faster models (CatVTON, Kling)
- Check browser console for detailed logs

## 📝 Project Structure

```
VTON/
├── public/
│   ├── index.html          # Main application
│   ├── app.js              # Application logic
│   └── config.js           # Configuration loader
├── server.py               # Python server (reads .env and serves API)
├── .env.example            # Environment variables template
├── .env                    # Your actual keys (NOT committed)
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── SETUP.md                # Setup instructions
├── package.json            # Project metadata
└── .github/
    ├── SECURITY.md         # Security policy
    └── CONTRIBUTING.md     # Contribution guide
```

## 🚀 Deployment

### Deploy to Vercel / Netlify

1. Push to GitHub
2. Connect repository to Vercel/Netlify
3. Set environment variables in dashboard (same as `.env.example`)
4. Deploy

### Deploy to Your Server

```bash
# Set environment variables
export FAL_KEY="your_key_id:your_key_secret"
export RP_EP="your_endpoint_id"
export RP_KEY="your_api_key"

# Run the server
python3 server.py
```

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit PRs.

## 📞 Support

- Issues or questions? Open a GitHub issue
- For API-specific help:
  - fal.ai: [docs.fal.ai](https://docs.fal.ai)
  - RunPod: [docs.runpod.io](https://docs.runpod.io)

---

**Powered by**: [fal.ai](https://fal.ai) · [FASHN](https://fashn.ai) · [RunPod](https://runpod.io)
