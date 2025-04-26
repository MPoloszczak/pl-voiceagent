# Programmable Voice Agent

## Overview

The **Programmable Voice Agent** is a multi-tenant, real-time voice application built with FastAPI that integrates with Twilio for call handling, Deepgram for Speech-to-Text, OpenAI for conversational intelligence and task execution, ElevenLabs for Text-to-Speech, and Redis for caching. It is designed to handle inbound phone calls, transcribe speech, process natural language interactions, and respond with high-quality synthesized audio.

## Features

- Real-time audio streaming via Twilio Voice Streams
- Speech-to-Text transcription with Deepgram
- Voice Activity Detection (VAD) for barge-in support
- AI-driven conversation flows with multiple OpenAI agents:
  - Triage Agent: routes conversation to the appropriate sub-agent
  - Booking Agent: collects appointment details and executes booking
  - General Conversation Agent: provides friendly, informative responses
- Dynamic, tenant-specific tool loading from an external MCP service
- Text-to-Speech response generation with ElevenLabs
- Multi-tenant database sessions (PostgreSQL with RLS and schema-per-tenant)
- Redis-based caching for temporary data storage
- AWS X-Ray instrumentation for end-to-end tracing
- Health checks and WebSocket metrics endpoints
- Docker and Docker Compose support for easy deployment
- NGROK integration for secure tunneling during local development

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Running the Application](#running-the-application)
4. [API Reference](#api-reference)
5. [Architecture Overview](#architecture-overview)
6. [Multi-Tenancy](#multi-tenancy)
7. [Caching](#caching)
8. [Logs & Monitoring](#logs--monitoring)
9. [Extending & Customizing](#extending--customizing)
10. [Project Structure](#project-structure)
11. [Contributing](#contributing)
12. [License](#license)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Redis server (for caching) or a running Redis instance
- PostgreSQL (optional) for multi-tenant storage (Aurora DSN)
- ngrok (optional) for tunneling during development
- Docker & Docker Compose (optional) for containerized setup

### Clone the Repository

```bash
git clone https://github.com/your-org/PL-VoiceAgent.git
cd PL-VoiceAgent
```

### Python Environment

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The application uses environment variables for configuration. Create a `.env` file or export the following variables:

| Variable              | Description                                                    | Required  |
|-----------------------|----------------------------------------------------------------|-----------|
| `DEEPGRAM_API_KEY`    | API key for Deepgram Speech-to-Text                            | Yes       |
| `ELEVENLABS_API_KEY`  | API key for ElevenLabs Text-to-Speech                          | Yes       |
| `OPENAI_API_KEY`      | API key for OpenAI GPT models                                  | Yes       |
| `REDIS_URL`           | Redis connection string (default: `redis://localhost:6379`)    | No        |
| `AURORA_DSN`          | PostgreSQL DSN for Aurora (default: local Postgres)            | No        |
| `MCP_BASE`            | Base URL for the MCP service (default: `http://mcp.local`)     | No        |
| `NGROK_URL`           | Public URL of ngrok tunnel (optional)                          | No        |
| `PORT`                | Port for the HTTP server (default: `8080`)                     | No        |
| `ENV`                 | Environment (`development` or `production`, affects reload)    | No        |


## Running the Application

### Local Development (FastAPI + Uvicorn)

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --reload
```

- The server will start on `http://localhost:8080`.
- Use `--reload` only in development mode.

### Docker & Docker Compose

1. Build and start services:
   ```bash
   docker-compose up --build
   ```
2. Services included:
   - **app**: the FastAPI voice agent
   - **redis**: caching layer
   - **db**: if configured, a PostgreSQL container (optional)
   - **ngrok**: for tunneling (optional)

Environment variables can be set in `docker-compose.yml` or via a `.env` file.

## API Reference

### HTTP Endpoints

- `GET /` 
  - Health check endpoint. 
  - Response: `{ "status": "healthy", "service": "Programmable Voice Agent" }`

- `GET /websocket-metrics`
  - Returns aggregate WebSocket connection metrics.
  - Response:
    ```json
    {
      "active_connections": <number>,
      "connection_attempts": <number>
    }
    ```

- `GET /websocket-status`
  - Returns detailed status of current WebSocket sessions.
  - Includes session IDs, client/application states, ngrok URL.

- `POST /voice`
  - Twilio Voice webhook endpoint to handle inbound calls.
  - Responds with TwiML instructing Twilio to open a WebSocket stream.
  - Requires TwiML response template defined in `utils.TWIML_STREAM_TEMPLATE`.

### WebSocket Endpoint

- `WS /twilio-stream`
  - Twilio will stream inbound audio frames over this WebSocket.
  - The server listens for media events, forwards to Deepgram, applies VAD, and streams synthesized audio back to Twilio.

## Architecture Overview

1. **Twilio Integration**: 
   - `twilio.py` implements `TwilioService` to handle call webhooks and audio WebSocket streams.
2. **Speech-to-Text**: 
   - `dpg.py` contains `DeepgramService` for live transcription.
3. **Voice Activity Detection**: 
   - `vad_events.py` uses WebRTC VAD to detect barge-in and manage turn-taking.
4. **Conversational AI**: 
   - `oai.py` defines OpenAI-powered agents (triage, booking, general) using `agents` framework.
   - Supports dynamic tools loaded per tenant via MCP client (`services/mcp_client.py`).
5. **Text-to-Speech**: 
   - `ell.py` implements `TTSService` for ElevenLabs synthesis and chunked audio streaming.
6. **Multi-Tenancy**: 
   - `middlewares/tenant.py` enforces an `X-Tenant-Id` header and scopes database sessions.
   - `db/session.py` manages `AsyncSession` with schema-per-tenant and RLS.
7. **Utilities**: 
   - `utils.py` provides logging, ngrok URL discovery, ping/keep-alive tasks.
8. **Caching**: 
   - `services/cache.py` implements JSON serialization in Redis.

## Multi-Tenancy

- All API requests must include `X-Tenant-Id` HTTP header.
- The middleware will reject missing or empty tenant IDs.
- Database sessions switch roles and search paths based on the tenant.
- Toolsets for OpenAI agents are loaded dynamically via MCP per tenant.

## Caching

- Redis is used to cache ephemeral data (e.g., streaming state, transient payloads).
- TTL defaults to 900 seconds and can be configured in `services/cache.py`.

## Logs & Monitoring

- Logging is configured in `utils.setup_logging`, writing to `logs/pl-voiceagent.log` and console.
- AWS X-Ray SDK (`aws_xray_sdk.core.patch_all()`) instruments standard libraries for distributed tracing.

## Extending & Customizing

- **Agents**: update prompts or tools in `oai.py` to change behavior.
- **TTS**: configure `TTSService` (voices, models) in `ell.py`.
- **VAD**: adjust thresholds and timeouts in `vad_events.py`.
- **MCP Tools**: provide a tenant-specific `toolbox.json` at the MCP endpoint to add custom tools.

## Project Structure

```
.
├── main.py                   # FastAPI application entrypoint
├── twilio.py                 # Twilio webhook and WebSocket service
├── dpg.py                    # Deepgram transcription service
├── ell.py                    # ElevenLabs TTS service
├── oai.py                    # OpenAI agent definitions and streaming
├── vad_events.py             # Voice Activity Detection for barge-in
├── utils.py                  # Logging, ngrok, ping/keep-alive utilities
├── middlewares/              # FastAPI middleware for tenant context
│   └── tenant.py
├── db/                       # Database session management (multi-tenant)
│   └── session.py
├── services/                 # Supporting service clients
│   ├── mcp_client.py         # Tenant-specific tool loader
│   └── cache.py              # Redis JSON cache
├── requirements.txt
├── dockerfile
└── docker-compose.yml
```

## Contributing

Contributions, issues, and feature requests are welcome! Please use GitHub Issues and pull requests to suggest changes.

## License

