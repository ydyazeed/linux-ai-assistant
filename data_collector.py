import os
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin

class LinuxDataCollector:
    def __init__(self):
        self.base_dir = Path("data")
        self.processed_dir = Path("processed_data")
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            # Command examples
            "command_examples/network_commands",
            "command_examples/security_commands",
            "command_examples/system_commands",
            # Error explanations
            "error_explanations/application_errors",
            "error_explanations/system_errors",
            "error_explanations/service_errors",
            # Network
            "network/configuration",
            "network/troubleshooting",
            "network/security",
            # Security
            "security/access_control",
            "security/encryption",
            "security/hardening"
        ]

        for dir_path in directories:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def collect_man_pages(self, commands: List[str], category: str):
        """Collect and process man pages for given commands"""
        for cmd in commands:
            try:
                # Get man page content
                result = subprocess.run(['man', cmd], capture_output=True, text=True)
                if result.returncode == 0:
                    content = result.stdout
                    # Process and save content
                    self._save_command_doc(cmd, content, category)
            except Exception as e:
                print(f"Error processing man page for {cmd}: {e}")

    def collect_arch_wiki(self, topic: str, category: str):
        """Collect data from Arch Wiki"""
        base_url = "https://wiki.archlinux.org"
        search_url = f"{base_url}/index.php?search={topic}&title=Special%3ASearch"
        
        try:
            response = requests.get(search_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find and process relevant articles
                for article in soup.select('.mw-search-result-heading a')[:5]:  # Top 5 results
                    if href := article.get('href'):
                        article_url = urljoin(base_url, href)
                        self._process_arch_wiki_article(article_url, category)
                        time.sleep(1)  # Be nice to their servers
        except Exception as e:
            print(f"Error collecting Arch Wiki data for {topic}: {e}")

    def collect_stack_exchange(self, tag: str, category: str):
        """Collect high-voted answers from Unix & Linux Stack Exchange"""
        api_url = "https://api.stackexchange.com/2.3/questions"
        params = {
            'site': 'unix.stackexchange.com',
            'tagged': tag,
            'sort': 'votes',
            'order': 'desc',
            'filter': 'withbody',
            'pagesize': 50
        }
        
        try:
            response = requests.get(api_url, params=params)
            if response.status_code == 200:
                data = response.json()
                self._process_stack_exchange_data(data['items'], category)
        except Exception as e:
            print(f"Error collecting Stack Exchange data for {tag}: {e}")

    def collect_distribution_docs(self, distro: str, topic: str, category: str):
        """Collect documentation from major distributions"""
        urls = {
            'ubuntu': f"https://help.ubuntu.com/community/{topic}",
            'rhel': f"https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/{topic}",
            'suse': f"https://documentation.suse.com/sles/15-SP4/html/SLES-all/article-{topic}.html"
        }
        
        if distro in urls:
            try:
                response = requests.get(urls[distro])
                if response.status_code == 200:
                    self._process_distro_doc(response.text, distro, category)
            except Exception as e:
                print(f"Error collecting {distro} docs for {topic}: {e}")

    def _save_command_doc(self, command: str, content: str, category: str):
        """Save processed command documentation"""
        filename = self.base_dir / category / f"{command}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    def _process_arch_wiki_article(self, url: str, category: str):
        """Process and save Arch Wiki article content"""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                if content_div := soup.select_one('#content'):
                    content = content_div.get_text()
                    
                    # Extract title from URL
                    title = url.split('/')[-1]
                    filename = self.base_dir / category / f"arch_wiki_{title}.txt"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
        except Exception as e:
            print(f"Error processing Arch Wiki article {url}: {e}")

    def _process_stack_exchange_data(self, items: List[dict], category: str):
        """Process and save Stack Exchange data"""
        for item in items:
            try:
                # Combine question and answers
                content = f"Question: {item['title']}\n\n{item['body']}"
                
                filename = self.base_dir / category / f"stack_exchange_{item['question_id']}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"Error processing Stack Exchange item {item.get('question_id')}: {e}")

    def _process_distro_doc(self, content: str, distro: str, category: str):
        """Process and save distribution documentation"""
        soup = BeautifulSoup(content, 'html.parser')
        # Extract main content area (varies by distribution)
        if main_content := soup.select_one('main, #content, .content'):
            filename = self.base_dir / category / f"{distro}_doc.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(main_content.get_text())

    def collect_all_data(self):
        """Collect all required data"""
        # Network Commands
        network_commands = [
            'ip', 'ifconfig', 'netstat', 'ss', 'ping', 'traceroute',
            'nslookup', 'dig', 'host', 'route', 'iptables', 'tcpdump'
        ]
        self.collect_man_pages(network_commands, "command_examples/network_commands")
        self.collect_arch_wiki("Networking", "network/configuration")
        self.collect_stack_exchange("networking", "network/troubleshooting")

        # Security Commands
        security_commands = [
            'chmod', 'chown', 'sudo', 'useradd', 'usermod', 'groupadd',
            'ssh-keygen', 'openssl', 'gpg', 'selinux', 'apparmor'
        ]
        self.collect_man_pages(security_commands, "command_examples/security_commands")
        self.collect_arch_wiki("Security", "security/hardening")
        self.collect_stack_exchange("security", "security/access_control")

        # System Commands
        system_commands = [
            'systemctl', 'journalctl', 'top', 'ps', 'df', 'du',
            'lsof', 'strace', 'ltrace', 'free', 'vmstat'
        ]
        self.collect_man_pages(system_commands, "command_examples/system_commands")
        
        # Application Errors
        for distro in ['ubuntu', 'rhel', 'suse']:
            self.collect_distribution_docs(distro, "Troubleshooting", "error_explanations/system_errors")
        self.collect_stack_exchange("error", "error_explanations/application_errors")

def main():
    collector = LinuxDataCollector()
    collector.collect_all_data()

if __name__ == "__main__":
    main() 