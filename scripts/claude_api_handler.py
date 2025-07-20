#!/usr/bin/env python3
"""
Claude API Handler for GitHub Actions
実際のClaude APIを呼び出してコード生成を行う
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from anthropic import Anthropic
from github import Github


class ClaudeCodeGenerator:
    def __init__(self, anthropic_key: str, github_token: str):
        self.anthropic = Anthropic(api_key=anthropic_key)
        self.github = Github(github_token)
        
    def analyze_issue(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """GitHubのIssueを分析"""
        repo = self.github.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        
        return {
            'number': issue.number,
            'title': issue.title,
            'body': issue.body,
            'labels': [label.name for label in issue.labels],
            'state': issue.state
        }
    
    def generate_implementation(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Claude APIを使用して実装を生成"""
        prompt = f"""
        You are tasked with implementing a GitHub issue for the segment-anything project.
        
        Issue #{issue_data['number']}: {issue_data['title']}
        
        Description:
        {issue_data['body']}
        
        Project Context:
        - This is a character extraction pipeline using YOLO + SAM
        - Follow the clean architecture: core/, features/, tests/, tools/
        - Use flake8, black, mypy, isort coding standards
        - Include comprehensive pytest tests
        
        Generate the implementation following this JSON structure:
        {{
            "files": [
                {{
                    "path": "features/processing/preprocessing/example.py",
                    "content": "# Python code here\\n..."
                }},
            ],
            "tests": [
                {{
                    "path": "tests/unit/test_example.py",
                    "content": "# Test code here\\n..."
                }}
            ],
            "summary": "Brief summary of what was implemented"
        }}
        
        Important:
        - Generate actual working Python code
        - Follow the existing project structure
        - Include proper error handling
        - Add type hints
        - Write comprehensive tests
        """
        
        message = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=8000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Parse Claude's response
        response_text = message.content[0].text
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                # Fallback: try to parse the entire response
                pass
        
        # If JSON parsing fails, create a structured response
        return {
            "files": [],
            "tests": [],
            "summary": "Failed to parse implementation",
            "raw_response": response_text
        }
    
    def create_files(self, implementation: Dict[str, Any]) -> List[str]:
        """生成されたファイルを作成"""
        created_files = []
        
        # Create implementation files
        for file_info in implementation.get('files', []):
            file_path = file_info['path']
            file_content = file_info['content']
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(file_content)
            
            created_files.append(file_path)
            print(f"✅ Created: {file_path}")
        
        # Create test files
        for test_info in implementation.get('tests', []):
            test_path = test_info['path']
            test_content = test_info['content']
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            
            # Write test file
            with open(test_path, 'w') as f:
                f.write(test_content)
            
            created_files.append(test_path)
            print(f"✅ Created test: {test_path}")
        
        return created_files
    
    def comment_on_issue(self, repo_name: str, issue_number: int, 
                        implementation: Dict[str, Any], created_files: List[str]):
        """Issueにコメントを追加"""
        repo = self.github.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        
        files_list = '\n'.join([f"- `{f}`" for f in created_files])
        
        comment = f"""
🤖 Claude has implemented the requested functionality!

**Summary**: {implementation.get('summary', 'Implementation completed')}

**Files created**:
{files_list}

**Next steps**:
1. Review the generated code
2. Run tests: `python -m pytest tests/ -v`
3. A pull request will be created automatically

Generated with Claude Code 🚀
"""
        
        issue.create_comment(comment)


def main():
    parser = argparse.ArgumentParser(description='Claude Code Generator for GitHub Issues')
    parser.add_argument('--repo', required=True, help='Repository name (owner/repo)')
    parser.add_argument('--issue', required=True, type=int, help='Issue number')
    parser.add_argument('--anthropic-key', help='Anthropic API key (or use env var)')
    parser.add_argument('--github-token', help='GitHub token (or use env var)')
    
    args = parser.parse_args()
    
    # Get API keys
    anthropic_key = args.anthropic_key or os.environ.get('ANTHROPIC_API_KEY')
    github_token = args.github_token or os.environ.get('GITHUB_TOKEN')
    
    if not anthropic_key or not github_token:
        print("❌ Error: API keys not found")
        sys.exit(1)
    
    # Initialize generator
    generator = ClaudeCodeGenerator(anthropic_key, github_token)
    
    try:
        # Analyze issue
        print(f"📋 Analyzing issue #{args.issue}...")
        issue_data = generator.analyze_issue(args.repo, args.issue)
        
        # Generate implementation
        print("🤖 Generating implementation with Claude...")
        implementation = generator.generate_implementation(issue_data)
        
        # Create files
        print("📝 Creating files...")
        created_files = generator.create_files(implementation)
        
        # Comment on issue
        print("💬 Commenting on issue...")
        generator.comment_on_issue(args.repo, args.issue, implementation, created_files)
        
        print("✅ Implementation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()