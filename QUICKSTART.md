# Quick Start Guide - AI YouTube Content Pipeline

## TL;DR - Get Started in 30 Seconds

```bash
cd /Users/divysingh/Downloads/template-python-main

# Option 1: Run the quick test script (recommended first time)
./run_pipeline_local.sh

# Option 2: Run the pipeline directly
./pipeline --config config.json

# Option 3: Run with explicit PYTHONPATH
PYTHONPATH=src python -m pipeline --config config.json
```

---

## Understanding the Commands

### Option 1: `./run_pipeline_local.sh` (Recommended First)

**What it does:**
- ✅ Checks your configuration
- ✅ Shows which APIs are configured vs using fallbacks
- ✅ Runs smoke tests to verify everything works
- ✅ Gives you a clear status report

**When to use:**
- First time running the pipeline
- After changing `.env` or `config.json`
- When troubleshooting issues

**Output:**
```
╔════════════════════════════════════════════════════════════════╗
║   AI YouTube Content Pipeline - Local Testing Mode            ║
╚════════════════════════════════════════════════════════════════╝

📋 Configuration Check
   • Project root: /Users/divysingh/Downloads/template-python-main
   • Config file: config.json ✓
   • Environment: .env ✓

🔑 API Key Status
   • ElevenLabs: ⚠ Not configured (will use placeholder MP3)
   • Claude: ✓ Configured (will use real API)
   • YouTube: ✓ Configured (will use real API)
   • Viewmax: ⚠ MCP not wired (will use placeholder clips)

✨ Running Smoke Tests...
✓ All tests passed!
```

---

### Option 2: `./pipeline --config config.json` (Quick Run)

**What it does:**
- Directly runs the pipeline
- Automatically sets up the Python path
- Uses the wrapper script for convenience

**When to use:**
- After initial testing is complete
- For regular pipeline runs
- When you know everything is configured

**Example:**
```bash
./pipeline --config config.json
```

---

### Option 3: `PYTHONPATH=src python -m pipeline --config config.json` (Manual)

**What it does:**
- Manually sets the Python path to include `src/`
- Runs the pipeline module directly
- Most explicit/verbose option

**When to use:**
- Debugging Python import issues
- Running from a different directory
- Integrating with other scripts

**Example:**
```bash
PYTHONPATH=src python -m pipeline --config config.json
```

---

## Why Do I Need PYTHONPATH?

The pipeline code is organized in a `src/` directory:
```
template-python-main/
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── __main__.py
│       ├── orchestrator/
│       ├── script_writer/
│       └── ...
├── config.json
└── .env
```

Python needs to know to look in `src/` to find the `pipeline` module. You can either:
1. Use the wrapper script `./pipeline` (handles this automatically)
2. Set `PYTHONPATH=src` manually before running

---

## Common Issues & Solutions

### Issue 1: "No module named pipeline"
```bash
python -m pipeline --config config.json
# Error: No module named pipeline
```

**Solution:** Use one of the methods above that sets PYTHONPATH:
```bash
# Use the wrapper
./pipeline --config config.json

# OR set PYTHONPATH manually
PYTHONPATH=src python -m pipeline --config config.json
```

---

### Issue 2: "Permission denied: ./pipeline"
```bash
./pipeline --config config.json
# Error: Permission denied
```

**Solution:** Make the script executable:
```bash
chmod +x ./pipeline
chmod +x ./run_pipeline_local.sh
```

---

### Issue 3: Pipeline hangs at "Awaiting Script Review"

This is **expected behavior**! The pipeline has two review gates:
- **Gate 1 (Script Review)**: After script generation
- **Gate 2 (Final Review)**: After all assets are generated

**What's happening:**
The pipeline is waiting for you to approve/edit the script through the Notion interface.

**Options:**
1. **Approve in Notion**: Open your Notion content calendar and approve the script
2. **Disable review gates**: Modify the orchestrator to skip gates for testing
3. **Let it timeout**: Gate 1 has a 72-hour timeout with reminders

---

### Issue 4: "File not found" errors for assets

```
Asset_Store url(...) failed: File not found
```

**This is normal** when using the local file-based Asset_Store. The pipeline is using a local JSON file instead of Google Drive.

**Expected behavior:**
- Files are stored in `pipeline_output/` directory
- Some stages might fail to find files if previous stages didn't complete
- This is fine for testing the fallback modes

---

## Testing Individual Stages

You can test specific stages without running the full pipeline:

```python
# Test narration generation
PYTHONPATH=src python -c "
import asyncio
from pipeline.narration_generator import ElevenLabsMCPClient

async def test():
    client = ElevenLabsMCPClient()
    mp3 = await client.synthesize(
        text='Test narration',
        voice_id='test',
        sample_rate=44_100,
        bitrate_kbps=128
    )
    print(f'Generated {len(mp3)} bytes')

asyncio.run(test())
"
```

---

## Understanding Fallback Modes

When you run the pipeline, you'll see log messages indicating which stages are using fallback mode:

```
[WARNING] ElevenLabs API key missing — using fallback mode (placeholder MP3)
[WARNING] YouTube credentials missing — using fallback mode (placeholder IDs)
[INFO] ViewmaxMCPClient initialized in fallback mode
```

**This is normal and expected!** It means:
- ✅ The pipeline can run without external APIs
- ✅ You'll get placeholder data instead of real media
- ✅ You can test the full workflow locally

See `PLACEHOLDER_FALLBACK_README.md` for details on what each fallback mode does.

---

## Next Steps

### 1. Run the Initial Test
```bash
./run_pipeline_local.sh
```

### 2. Review the Output
Check that tests pass and review which services are using fallbacks.

### 3. (Optional) Configure Real APIs
If you want real narration, videos, or uploads, add API keys to `.env`:

```bash
# For ElevenLabs narration
ELEVENLABS_API_KEY=your_real_key

# For YouTube uploads (already configured in your .env)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REFRESH_TOKEN=...
```

### 4. Run the Full Pipeline
```bash
./pipeline --config config.json
```

---

## Useful Commands Reference

```bash
# Quick test everything
./run_pipeline_local.sh

# Run pipeline (wrapper script)
./pipeline --config config.json

# Run pipeline (manual PYTHONPATH)
PYTHONPATH=src python -m pipeline --config config.json

# Run just the smoke tests
PYTHONPATH=src python test_pipeline_local.py

# Check Python imports
PYTHONPATH=src python -c "from pipeline.factory import build_orchestrator; print('OK')"

# View pipeline help
./pipeline --help

# Check what's in your calendar
cat pipeline_output/calendar.json | python -m json.tool
```

---

## Getting Help

1. **Check logs**: Look for detailed error messages in console output
2. **Review docs**: 
   - `PLACEHOLDER_FALLBACK_README.md` - Fallback mode details
   - `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
3. **Test stages individually**: Use the test script to isolate issues

---

## Summary

**To get started right now:**
```bash
cd /Users/divysingh/Downloads/template-python-main
./run_pipeline_local.sh
```

That's it! The script will test everything and tell you if the pipeline is ready to run. 🚀
