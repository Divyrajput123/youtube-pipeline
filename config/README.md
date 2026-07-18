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

## Security Notes

- **Never commit `config.json` or `.env` to version control**
- These files contain sensitive API keys and tokens
- The `.gitignore` file is configured to exclude them
- Always use the example files as templates
