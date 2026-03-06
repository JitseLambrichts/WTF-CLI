# WTF-CLI (What The Fix CLI)


![Rust](https://img.shields.io/badge/language-Rust-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Issues](https://img.shields.io/github/issues/JitseLambrichts/WTF-CLI)
![Last Commit](https://img.shields.io/github/last-commit/JitseLambrichts/WTF-CLI)
![Stars](https://img.shields.io/github/stars/JitseLambrichts/WTF-CLI)

`wtf-cli` is a command-line interface wrapper that seamlessly runs your terminal commands and, if they fail, intercepts the error output to provide an AI-generated solution on the spot. It supports local models via **Ollama**, as well as cloud-based ones via **OpenAI**, **Gemini**, or **OpenRouter**.

<p align="center">
    <img src="https://github.com/user-attachments/assets/9b4d7f21-2906-4342-835b-22078205d2b4" width=70%/>
<p>

## Features

- **Seamless wrapping**: Just prepend `wtf` to your command. If it works, you get your normal output. If it fails, the AI jumps in.
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
   git clone https://github.com/yourusername/wtf-cli.git
   cd wtf-cli
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

4. Configure your preferred AI provider:

   ```bash
   wtf --setup
   ```

## Updating

If you've installed `wtf-cli` from source and want to pull the latest changes:

1. Navigate to your local `wtf-cli` repository:
   ```bash
   cd path/to/wtf-cli
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

Simply prepend `wtf` to any command you usually run.

```bash
# Example 1: A failing npm script
wtf npm run build

# Example 2: Exploring a non-existent directory
wtf ls /fake/directory
```

If the command succeeds, it will gracefully exit just like normally.
If it fails, `wtf` will capture the error output, send it to the configured AI, and print the diagnosis and suggested fix back to you.

## Configuration

You can easily configure your preferred AI provider by running `wtf --setup`. This command will present an interactive menu allowing you to choose between Ollama, OpenAI, Gemini, and OpenRouter using your arrow keys. It will automatically create or update a `.env` file in the current directory with your selection.

Alternatively, you can manually create a `.env` file in the directory where you run the tool. A template is provided in `.env.example`:

```bash
cp .env.example .env
```

Or set these Environment Variables globally:

```env
# AI Provider (auto-detected if not set)
# Options: ollama, openai, gemini, openrouter
WTF_PROVIDER=ollama

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

## Demo
<video src="https://github.com/user-attachments/assets/04c37c5a-e0f8-4287-ac59-6304b995f947" controls width="600">
  Je browser ondersteunt geen video.
</video>


## License

MIT
