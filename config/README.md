# Configuration Directory

This directory contains example configuration files. Copy these to the project root and customize them with your actual values.

## Files

### `config.example.json`
Main pipeline configuration file. Copy to `../config.json` and customize.

### `.env.example`
Environment variables for API keys and secrets. Copy to `../.env` and add your actual keys.

## Setup Instructions

```bash
# From the project root
cp config/config.example.json config.json
cp config/.env.example .env

# Edit the files with your actual values
nano config.json
nano .env
```

## Video Provider Toggle

Set `visual_video_provider` in `config.json` to `"kling"` or `"runpod"`. Keep
both provider credentials in `.env` if you want to switch later; only the
selected provider is called. The default is `"kling"`.

## Security Notes

- **Never commit `config.json` or `.env` to version control**
- These files contain sensitive API keys and tokens
- The `.gitignore` file is configured to exclude them
- Always use the example files as templates
