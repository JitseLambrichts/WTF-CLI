# WTH-CLI (What The Heck CLI)

![Rust](https://img.shields.io/badge/language-Rust-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Stars](https://img.shields.io/github/stars/JitseLambrichts/WTH-CLI)
![Issues](https://img.shields.io/github/issues/JitseLambrichts/WTH-CLI)
![Last Commit](https://img.shields.io/github/last-commit/JitseLambrichts/WTH-CLI)

`wth-cli` is a command-line interface wrapper that seamlessly runs your terminal commands and, if they fail, intercepts the error output to provide an AI-generated solution on the spot. It supports local models via **Ollama**, as well as cloud-based ones via **OpenAI**, **Gemini**, or **OpenRouter**.

## Features

- **Seamless wrapping**: Just prepend `wth` to your command. If it works, you get your normal output. If it fails, the AI jumps in.
- **Privacy first**: The primary focus is running local AI models using [Ollama](https://ollama.com/), meaning no API costs and total privacy.
- **Cloud Fallbacks**: Supports OpenAI (`OPENAI_API_KEY`), Gemini (`GEMINI_API_KEY`), and OpenRouter (`OPENROUTER_API_KEY`) fallbacks.
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

3. Ensure the Cargo bin directory is in your system's `PATH`. You can copy these commands exactly; your shell will automatically expand variables like `$HOME` or `$env:USERPROFILE`.

   **Linux / macOS (Bash/Zsh):**

   ```bash
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

   _Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent._

   **Windows (PowerShell):**

   ```powershell
   $env:Path += ";$env:USERPROFILE\.cargo\bin"
   ```

   _To make this permanent, add it to your [PowerShell Profile](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_profiles) or use the 'Environment Variables' GUI._

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

# Gemini Fallback
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.0-flash

# OpenRouter Fallback
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=arcee-ai/trinity-mini:free
```

## License

MIT
