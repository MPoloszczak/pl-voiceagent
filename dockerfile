# Use Python 3.11 slim image for efficiency
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing and AWS X-Ray
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables with defaults
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Environment variables that must be configured at runtime:
# - OPENAI_API_KEY: Required for OpenAI agent functionality
# - DEEPGRAM_API_KEY: Required for speech-to-text transcription  
# - ELEVENLABS_API_KEY: Required for text-to-speech synthesis
# - MCP_SERVER_URL: URL for MCP server connection (default: http://localhost:8080/mcp)
# - REDIS_URL: Required for conversation history caching
# - Other optional environment variables documented in README

# Expose the application port
EXPOSE 8000

# Add health check with timeout and retries
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"] 

