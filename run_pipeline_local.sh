#!/bin/bash
# Quick script to run the AI YouTube Content Pipeline locally with placeholder fallbacks
# Usage: ./run_pipeline_local.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   AI YouTube Content Pipeline - Local Testing Mode            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "config.json" ]; then
    echo "❌ Error: config.json not found. Please run from project root."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please run from project root."
    exit 1
fi

echo -e "${YELLOW}📋 Configuration Check${NC}"
echo "   • Project root: $(pwd)"
echo "   • Config file: config.json ✓"
echo "   • Environment: .env ✓"
echo ""

# Check which services are configured
echo -e "${YELLOW}🔑 API Key Status${NC}"

# Check ElevenLabs
if grep -q "^ELEVENLABS_API_KEY=sk_" .env 2>/dev/null; then
    echo "   • ElevenLabs: ✓ Configured (will use real API)"
else
    echo "   • ElevenLabs: ⚠ Not configured (will use placeholder MP3)"
fi

# Check Anthropic/Claude
if grep -q "^ANTHROPIC_API_KEY=sk-ant-" .env 2>/dev/null; then
    echo "   • Claude: ✓ Configured (will use real API)"
else
    echo "   • Claude: ⚠ Not configured (will use placeholders)"
fi

# Check YouTube
if grep -q "^GOOGLE_CLIENT_ID=" .env 2>/dev/null && \
   grep -q "^GOOGLE_CLIENT_SECRET=" .env 2>/dev/null && \
   grep -q "^GOOGLE_REFRESH_TOKEN=" .env 2>/dev/null; then
    echo "   • YouTube: ✓ Configured (will use real API)"
else
    echo "   • YouTube: ⚠ Not configured (will use placeholder IDs)"
fi

# Viewmax is always placeholder for now
echo "   • Viewmax: ⚠ MCP not wired (will use placeholder clips)"

echo ""
echo -e "${GREEN}✨ Running Smoke Tests First...${NC}"
echo ""

# Run tests
export PYTHONPATH="$(pwd)/src"
if python test_pipeline_local.py; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
else
    echo ""
    echo "❌ Tests failed. Please check the output above."
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Pipeline is ready!                                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now run the full pipeline with:"
echo ""
echo "  python -m pipeline --config config.json"
echo ""
echo "Or test individual stages programmatically."
echo ""
echo -e "${YELLOW}💡 Tip:${NC} Check PLACEHOLDER_FALLBACK_README.md for detailed usage guide"
echo ""
