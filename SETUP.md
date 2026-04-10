# Setup Guide - Step by Step

Complete guide to setting up Virtual Try-On locally and preparing for deployment.

## Prerequisites

- **Python 3.6+** (for local server)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **Git** (for version control)
- Internet connection (for API calls)

## Local Setup (Development)

### Step 1: Clone or Download Repository

```bash
git clone https://github.com/yourusername/VTON.git
cd VTON
```

Or download and extract the ZIP file.

### Step 2: Get API Keys

#### FAL.AI (Required for most models)

1. Go to [fal.ai](https://fal.ai)
2. Sign up for free account
3. Go to [Dashboard → API Keys](https://fal.ai/dashboard/keys)
4. Copy your key in format: `key_id:key_secret`

#### RunPod (Optional - only for FASHN model)

1. Go to [RunPod](https://www.runpod.io)
2. Create account and set up GPU endpoint
3. Get [API Key from Console](https://www.runpod.io/console/api-keys)
4. Note your endpoint ID

### Step 3: Configure Environment

1. Copy template file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your keys:

   ```bash
   # On Windows (PowerShell)
   notepad .env

   # On Mac/Linux
   nano .env
   ```

3. Add your keys:

   ```
   FAL_KEY=your_key_id:your_key_secret
   RP_EP=your_endpoint_id
   RP_KEY=your_api_key
   ```

4. Save and close

### Step 4: Start Local Server

**Windows (PowerShell):**

```powershell
cd VTON
python server.py
```

**Mac/Linux:**

```bash
cd VTON
python3 server.py
```

You should see:

```
Virtual Try-On Server
====================
Port: 8080
URL: http://localhost:8080/public/

Environment Variables Loaded:
  FAL_KEY: ✓
  RP_EP: ✓ (optional)
  RP_KEY: ✓ (optional)
```

### Step 5: Open in Browser

Visit: [http://localhost:8080/public/](http://localhost:8080/public/)

### Step 6: Test the App

1. Upload a portrait photo (Person)
2. Upload a garment image (Garment)
3. Select a model
4. Click "Try On"
5. Download the result

## Troubleshooting

### API Key Issues

**Error: "Set FAL_KEY"**

- Check `.env` file exists
- Verify key format: `id:secret` with colon separator
- Ensure no extra spaces

**Error: "CORS blocked"**

- Don't open file directly (file://)
- Always use http://localhost:8080

### Image Upload Issues

- Use clear, well-lit photos
- Recommend 512x512 or larger
- JPEG or PNG format
- Full-body for person images

### Slow Processing

- Model is processing (normal 30-120s)
- Check console logs for status
- Try models in order: CatVTON → Kling → Others
- Check API status pages

## Deployment

### Deploy to Vercel

1. Push to GitHub
2. Go to [Vercel](https://vercel.com)
3. Import GitHub repository
4. Add environment variables:
   - `FAL_KEY`
   - `RP_EP`
   - `RP_KEY`
5. Deploy

### Deploy to Netlify

1. Push to GitHub
2. Go to [Netlify](https://netlify.com)
3. Connect GitHub repo
4. Add environment variables in dashboard
5. Deploy

### Self-Hosted Server

```bash
# Install Python if needed
sudo apt-get install python3

# Clone repo
git clone https://github.com/yourusername/VTON.git
cd VTON

# Set environment variables
export FAL_KEY="your_key_id:your_key_secret"
export RP_EP="your_endpoint"
export RP_KEY="your_key"

# Start server
python3 server.py
```

## Project Structure

```
VTON/
├── public/
│   ├── index.html      # Main app
│   ├── app.js          # Application logic
│   └── config.js       # Configuration loader
├── .env.example        # Environment template
├── .env                # Your actual keys (DON'T COMMIT)
├── .gitignore          # Git ignore rules
├── README.md           # User documentation
├── SETUP.md            # This file
├── package.json        # Project metadata
└── .github/
    ├── SECURITY.md     # Security policy
    └── CONTRIBUTING.md # Contribution guide
```

## Next Steps

1. ✅ Clone/download repository
2. ✅ Get API keys from fal.ai
3. ✅ Create `.env` file with keys
4. ✅ Start local server
5. ✅ Test in browser
6. ✅ (Optional) Deploy to cloud

## Support

- Check [README.md](README.md) for features
- Review [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for contributing
- Open GitHub issues for bugs
- Check [.github/SECURITY.md](.github/SECURITY.md) for security concerns

## API Documentation

- **FAL.AI**: [docs.fal.ai](https://docs.fal.ai)
- **RunPod**: [docs.runpod.io](https://docs.runpod.io)
- **CatVTON**: [Model API](https://fal.ai/models/fal-ai/cat-vton/api)

---

Happy trying on! 🎉
