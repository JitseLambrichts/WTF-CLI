# WTH-CLI (What The Heck CLI)

`wth-cli` is a command-line interface wrapper that seamlessly runs your terminal commands and, if they fail, intercepts the error output to provide an AI-generated solution on the spot. It supports local models via **Ollama**, as well as cloud-based ones via **OpenAI** or **OpenRouter**.

Written in blazingly fast Rust.

## Features

- **Seamless wrapping**: Just prepend `wth` to your command. If it works, you get your normal output. If it fails, the AI jumps in.
- **Privacy first**: The primary focus is running local AI models using [Ollama](https://ollama.com/), meaning no API costs and total privacy.
- **Cloud Fallbacks**: Supports OpenAI (`OPENAI_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`) fallbacks.
- **Clear structure**: Provides actionable, structured outputs so you know exactly what failed and the command to fix it.

## Prerequisites

- [Rust & Cargo](https://rustup.rs/) (latest stable version recommended)
- Optional (but recommended): [Ollama](https://ollama.com/) running locally for free, private AI analysis.

## Installation

### From Source

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/wth-cli.git
   cd wth-cli
   ```

2. Install the binary using Cargo:
   ```bash
   cargo install --path .
   ```
   _Make sure `~/.cargo/bin` is in your system's `PATH`._

## Updating

If you've installed `wth-cli` from source and want to pull the latest changes:

1. Navigate to your local `wth-cli` repository:
   ```bash
   cd path/to/wth-cli
   ```
2. Pull the latest code:
   ```bash
   git pull origin main
   ```
3. Re-install the project (the `--force` flag ensures the old binary gets overwritten):
   ```bash
   cargo install --force --path .
   ```

## Usage

Simply prepend `wth` to any command you usually run.

```bash
# Example 1: A failing npm script
wth npm run build

# Example 2: Exploring a non-existent directory
wth ls /fake/directory
```

If the command succeeds, it will gracefully exit just like normally.
If it fails, `wth` will capture the error output, send it to the configured AI, and print the diagnosis and suggested fix back to you.

## Configuration

`wth-cli` attempts to connect to local Ollama by default, but it can be customized.
You can create a `.env` file in the directory where you run the tool. A template is provided in `.env.example`:

```bash
cp .env.example .env
```

Or set these Environment Variables globally:

```env
# Ollama (Default provider)
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_HOST=http://localhost:11434

# OpenAI Fallback
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
# OPENAI_API_BASE=https://api.openai.com/v1

# OpenRouter Fallback
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=arcee-ai/trinity-mini:free
```

## License

MIT
