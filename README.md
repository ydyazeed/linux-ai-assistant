# Linux AI ChatBot (NUDU)

NUDU (Native Unix Diagnostic Utility) is your personal Linux command-line assistant that runs entirely on your machine. Ask questions about Linux commands, system administration, or troubleshooting in plain English, and get helpful, contextual answers - all while staying offline and secure.

## Why Choose This Assistant?

- **100% Offline & Private**: All processing happens on your machine. No data ever leaves your system.
- **Natural Conversations**: Ask questions in plain English - no need to remember exact command syntax.
- **Always Available**: Works without internet connection, perfect for server maintenance or network issues.
- **Comprehensive Knowledge**: Draws from Linux manual pages and common troubleshooting guides.
- **Terminal Integration**: Seamlessly integrates with your command line workflow.

## Perfect For

### System Administrators
- Quick troubleshooting of system issues without leaving the terminal
- Instant access to best practices for system configuration and security
- Faster resolution of common Linux problems and error messages
- Help with performance tuning and system optimization
- Guidance on backup strategies and disaster recovery

### DevOps Engineers
- Assistance with CI/CD pipeline configuration and debugging
- Help with container orchestration and deployment issues
- Quick reference for infrastructure-as-code best practices
- Support for automation script development
- Guidance on monitoring and logging setup

### Daily Operations
- Server maintenance and health checks
- Security auditing and hardening
- Network configuration and troubleshooting
- Service deployment and management
- Log analysis and system monitoring
- Performance optimization and resource management

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- ~5GB disk space for the model and embeddings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LinuxAIChatBot.git
cd LinuxAIChatBot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required files:
```bash
# Create necessary directories
mkdir -p processed_data data

# Download the Mistral-7B model (automatically downloaded on first run, or manually):
wget -O mistral-7b-instruct-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Download pre-computed embeddings and indices:
wget -P processed_data/ https://huggingface.co/datasets/ydyazeed/linux-ai-chatbot-embedding/resolve/main/linux_manual_embeddings.npy
wget -P processed_data/ https://huggingface.co/datasets/ydyazeed/linux-ai-chatbot-embedding/resolve/main/linux_manual_faiss.index
wget -P processed_data/ https://huggingface.co/datasets/ydyazeed/linux-ai-chatbot-embedding/resolve/main/linux_manual_metadata.pkl
wget -P processed_data/ https://huggingface.co/datasets/ydyazeed/linux-ai-chatbot-embedding/resolve/main/common_issues_embeddings.npy
wget -P processed_data/ https://huggingface.co/datasets/ydyazeed/linux-ai-chatbot-embedding/resolve/main/common_issues_faiss.index
wget -P processed_data/ https://huggingface.co/datasets/ydyazeed/linux-ai-chatbot-embedding/resolve/main/common_issues_metadata.pkl

# Or use the provided download script (recommended):
python download_embeddings.py
```

5. Make the command-line interface executable:
```bash
chmod +x nudu
```

6. (Optional) Add to your PATH for system-wide access:
```bash
# Add this line to your ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/LinuxAIChatBot"
```

## Usage

1. Start the assistant server in a separate terminal:
```bash
python chatbot_server.py
```

2. Ask questions using the `nudu` command:
```bash
# Basic questions
nudu "how do I check disk usage in Linux?"
nudu "what's the difference between sudo and su?"
nudu "explain chmod command"

# More complex queries
nudu "how can I find and delete files larger than 1GB that haven't been accessed in 30 days?"
nudu "what are the best practices for securing SSH access?"
nudu "troubleshoot slow network performance on Ubuntu"

# Getting help
nudu --help
```

The assistant will provide detailed responses with explanations, examples, and best practices based on Linux documentation.

## Tips for Better Results

1. **Be Specific**: Instead of "fix error", try "fix Permission denied error when running ./script.sh"
2. **Provide Context**: Mention your Linux distribution or specific environment when relevant
3. **Ask Follow-ups**: If you need clarification, ask follow-up questions about the previous answer
4. **Request Examples**: Add "show example" or "with examples" to see practical usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mistral AI for the Mistral-7B model
- The Linux community for comprehensive documentation
- All contributors and users of this project

## Note

The model and knowledge base files are not included in the repository due to their size. They will be downloaded during installation using the provided commands. 