FROM python:3.12-slim

# Install system dependencies for C extensions and audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    portaudio19-dev \
    libasound-dev \
    libssl-dev \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install Python packages with optimized compiler settings
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default environment variables
ENV PORT=8080
ENV ENV=production

# These environment variables should be passed at runtime or using
# docker-compose.yml with a .env file for security.
# You can use the following command to run with environment variables:
# docker run -p 443:443 --env-file .env voiceagent
# 
# Required environment variables include:
# - OPENAI_API_KEY
# - TWILIO_ACCOUNT_SID
# - TWILIO_AUTH_TOKEN
# - TWILIO_PHONE_NUMBER
# - ELEVENLABS_API_KEY
# - DEEPGRAM_API_KEY

CMD ["python", "main.py"] 

