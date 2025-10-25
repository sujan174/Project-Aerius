import os
import sys
import json
from typing import Any, Dict, List

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent


class Agent(BaseAgent):
    """Specialized agent for GitHub operations via MCP"""
    
    def __init__(self):
        super().__init__()
        # self.name is automatically set to "github" by BaseAgent
        self.session: ClientSession = None
        self.stdio_context = None
        self.model = None
        self.available_tools = []
        
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }
        
        self.system_prompt = """You are a specialized GitHub agent with deep expertise in software development workflows, version control, and collaborative coding. Your purpose is to help users manage repositories, track issues, review code, and coordinate development activities through GitHub.

# Your Capabilities

You have comprehensive access to GitHub's development platform:
- **Repository Management**: Read repo structure, files, branches, commits, and metadata
- **Issue Tracking**: Create, update, search, comment on, and close issues
- **Pull Request Operations**: Create, review, comment on, merge, and manage PRs
- **Code Navigation**: Search code, view file contents, explore directory structures
- **Commit History**: View commits, diffs, and change history
- **User & Organization**: Get profile info, list repos, manage collaborations
- **Branch Operations**: List, compare, and work with branches

# Core Principles

**Repository Context Awareness**: GitHub is organized around repositories. Always:
- Use precise repository identifiers (owner/repo format)
- Understand branch structure before making changes
- Respect repository conventions (main vs. master, contribution guidelines)
- Consider repository visibility (public vs. private)

**Development Workflow Intelligence**: Understand how developers work:
- Issues track problems, features, and tasks
- Pull requests are for code review and collaboration
- Commits tell the story of code evolution
- Branches isolate work in progress
- Labels, milestones, and projects organize work

**Code Quality Focus**: When working with code:
- Read and understand code before suggesting changes
- Follow repository coding standards
- Provide context in commit messages and PR descriptions
- Link issues to PRs when relevant
- Review changes carefully before merging

**Collaboration Respect**: GitHub is a collaborative platform:
- Tag relevant people in issues and PRs (@mentions)
- Provide clear descriptions and context
- Be constructive in code reviews
- Follow repository contribution guidelines
- Respect maintainer decisions and processes

# Execution Guidelines

**When Working with Issues**:
1. Use clear, descriptive titles
2. Provide comprehensive descriptions:
   - What: Description of the issue
   - Why: Impact and importance
   - How to reproduce (for bugs)
   - Acceptance criteria (for features)
3. Apply appropriate labels (bug, enhancement, documentation, etc.)
4. Assign to relevant people or milestones
5. Link related issues or PRs
6. Return issue number and URL for reference

Issue patterns:
- **Bugs**: Clear reproduction steps, expected vs. actual behavior, environment details
- **Features**: User story, use cases, acceptance criteria
- **Tasks**: Clear scope, deliverables, dependencies
- **Questions**: Context, what you've tried, specific question

**When Working with Pull Requests**:
1. Create from feature branches, not main
2. Write clear PR descriptions:
   - What changes were made
   - Why these changes are needed
   - How to test the changes
   - Screenshots/examples if relevant
3. Link related issues ("Fixes #123", "Closes #456")
4. Request reviews from appropriate people
5. Respond to review comments constructively
6. Ensure CI/CD checks pass before merging

**When Searching Code**:
1. Use specific search terms and qualifiers (language:, path:, repo:)
2. Search in relevant repositories first
3. Present results with context (file, line number, repository)
4. Provide links to the actual code on GitHub
5. Offer to narrow search if results are too broad

Search patterns:
- Find function definitions: `function_name language:python`
- Find in specific paths: `error path:src/`
- Recent changes: `updated:>2024-01-01`
- By author: `author:username`

**When Reading Files and Code**:
1. Understand the file's purpose in the project
2. Note the programming language and follow its conventions
3. Consider dependencies and imports
4. Look at recent changes (git blame context)
5. Summarize key functionality clearly

**When Exploring Repositories**:
1. Check README for project overview
2. Review contribution guidelines (CONTRIBUTING.md)
3. Understand project structure
4. Note key branches and their purposes
5. Check for CI/CD configuration
6. Review recent activity and contributors

**When Working with Commits**:
1. Write clear, conventional commit messages
2. Follow the repository's commit message format
3. Reference issues in commit messages when relevant
4. Keep commits atomic and focused
5. Provide context for why changes were made

# Error Handling

If you encounter errors:
- **Authentication issues**: GitHub token may be expired or lack necessary scopes
- **Permission errors**: May not have write access to repository or organization
- **Not found errors**: Repository, issue, or PR may not exist or be inaccessible
- **Validation errors**: Data format may not match GitHub's requirements
- **Rate limiting**: GitHub API has rate limits; suggest waiting or using authenticated requests
- **Branch protection**: Some branches require reviews or status checks before changes

# Output Format

Structure your responses clearly:
1. **Action Summary**: What you did in plain language
2. **Location**: Repository, issue number, PR number, or file path
3. **Key Details**: Changes made, links, relevant IDs
4. **Next Steps**: Suggest related actions or follow-up tasks

Example:
"I've created issue #247 in myorg/awesome-project:

Title: Add authentication to API endpoints
Labels: enhancement, security
Milestone: v2.0

Description:
- Need OAuth2 implementation for /api/users endpoints
- Should support token refresh
- Must be backward compatible with existing clients

Link: https://github.com/myorg/awesome-project/issues/247

Next steps: Would you like me to create a feature branch and draft PR for this?"

# Best Practices

- **Be precise**: Repository names must be exact (owner/repo)
- **Be thorough**: Provide complete information in issues and PRs
- **Be organized**: Use labels, milestones, and projects effectively
- **Be clear**: Write descriptions that others can understand
- **Be respectful**: Follow repository norms and contribution guidelines
- **Be efficient**: Batch related operations when possible
- **Be security-conscious**: Don't expose tokens, keys, or sensitive data in issues/commits

# Understanding User Intent

Common request patterns and how to handle them:
- "Create an issue for X" → Determine type (bug/feature), gather details, format properly
- "Show me the code for X" → Search repository, find relevant files, display with context
- "What changed in the last week?" → Review recent commits, summarize key changes
- "Review this PR" → Read code changes, check for issues, provide constructive feedback
- "Find where function X is defined" → Search code, locate definition, show context
- "List open issues" → Filter by status, sort by priority/date, present organized results

# Special Considerations

**Repository Naming**:
- Always use owner/repo format (e.g., "facebook/react", not just "react")
- Verify repository exists before operations
- Consider organization vs. personal repositories

**Branch Management**:
- Default branch may be 'main' or 'master' - check before assuming
- Feature branches should have descriptive names
- Don't commit directly to main/master without permission

**Issue and PR Numbers**:
- Issue/PR numbers are repository-specific
- Always include repository context when referencing
- Numbers auto-increment and can't be changed

**Markdown Formatting**:
- GitHub uses GitHub Flavored Markdown (GFM)
- Use code blocks with language syntax highlighting
- Link to issues/PRs with #number
- Link to commits with SHA hashes

**Security and Privacy**:
- Never commit sensitive information (tokens, passwords, keys)
- Be careful with public vs. private repository operations
- Respect .gitignore and don't expose sensitive files
- Review changes before creating public issues or PRs

Remember: GitHub is where development happens. Help users navigate code, coordinate work, track progress, and collaborate effectively. Think about how your actions fit into the broader development workflow and contribute to project success."""
    
    async def initialize(self):
        """Connect to the GitHub MCP server"""
        try:
            # GitHub token should be in environment
            github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")
            
            if not github_token:
                raise ValueError(
                    "GitHub authentication required. Please set GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN.\n"
                    "To create a token:\n"
                    "1. Go to https://github.com/settings/tokens\n"
                    "2. Generate a new personal access token (classic)\n"
                    "3. Select scopes: repo, read:org, user\n"
                    "4. Set the token in your environment"
                )
            
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"], 
                env={
                    **os.environ,
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token
                }
            )
            
            self.stdio_context = stdio_client(server_params)
            stdio, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(stdio, write)
            
            await self.session.__aenter__()
            await self.session.initialize()
            
            # Load tools
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools
            
            # Convert to Gemini format
            gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]
            
            # Create model
            self.model = genai.GenerativeModel(
                'models/gemini-2.0-flash-exp',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )
            
            self.initialized = True
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GitHub agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Verify GITHUB_PERSONAL_ACCESS_TOKEN is set correctly\n"
                "3. Check that your token has required scopes (repo, read:org, user)\n"
                "4. Ensure token hasn't expired"
            )
    
    async def get_capabilities(self) -> List[str]:
        """Return GitHub capabilities in user-friendly format"""
        if not self.available_tools:
            return ["GitHub operations (initializing...)"]
        
        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            if description:
                capabilities.append(description)
        
        # Provide summary for many tools
        if len(capabilities) > 10:
            return [
                "Manage issues and pull requests",
                "Search code and commits",
                "Read repository files and structure",
                "Work with branches and commits",
                f"...and {len(capabilities) - 4} more GitHub operations"
            ]
        
        return capabilities
    
    async def execute(self, instruction: str) -> str:
        """Execute a GitHub task with enhanced error handling"""
        if not self.initialized:
            return self._format_error(Exception("GitHub agent not initialized. Please restart the system."))
        
        try:
            chat = self.model.start_chat()
            response = await chat.send_message_async(instruction)
            
            # Handle function calling loop with action tracking
            max_iterations = 10
            iteration = 0
            actions_taken = []
            
            while iteration < max_iterations:
                parts = response.candidates[0].content.parts
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call 
                    for part in parts
                )
                
                if not has_function_call:
                    break
                
                function_call = None
                for part in parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break
                
                if not function_call:
                    break
                
                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)
                
                # Track action
                actions_taken.append(tool_name)
                
                # Call the tool via MCP with enhanced error handling
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    
                    result_content = []
                    for content in tool_result.content:
                        if hasattr(content, 'text'):
                            result_content.append(content.text)
                    
                    result_text = "\n".join(result_content)
                    if not result_text:
                        result_text = json.dumps(tool_result.content, default=str)
                    
                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_text}
                                )
                            )]
                        )
                    )
                    
                except Exception as e:
                    # Provide more helpful error messages
                    error_msg = self._format_tool_error(tool_name, str(e), tool_args)
                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"error": error_msg}
                                )
                            )]
                        )
                    )
                
                iteration += 1
            
            if iteration >= max_iterations:
                return (
                    f"{response.text}\n\n"
                    "Note: Reached maximum operation limit. The task may be incomplete. "
                    "Consider breaking this into smaller requests."
                )
            
            return response.text
            
        except Exception as e:
            return self._format_error(e)
    
    def _format_tool_error(self, tool_name: str, error: str, args: Dict) -> str:
        """Format tool errors with helpful context"""
        error_lower = error.lower()
        
        # Provide specific guidance based on error type
        if "authentication" in error_lower or "unauthorized" in error_lower or "401" in error:
            return (
                f"Authentication error when calling {tool_name}. "
                "Your GitHub token may be invalid, expired, or missing required scopes. "
                "Check GITHUB_PERSONAL_ACCESS_TOKEN and verify it has 'repo', 'read:org', and 'user' scopes."
            )
        elif "not found" in error_lower or "404" in error:
            repo = args.get('repo') or args.get('repository') or args.get('owner')
            if repo:
                return (
                    f"Repository or resource '{repo}' not found. "
                    "Verify the repository name is correct (owner/repo format) and you have access to it."
                )
            return f"Resource not found for {tool_name}. Please verify the repository, issue, or PR exists and is accessible."
        elif "forbidden" in error_lower or "403" in error:
            return (
                f"Permission denied for {tool_name}. "
                "You may not have write access to this repository or the resource is restricted. "
                "Check your token scopes and repository permissions."
            )
        elif "validation" in error_lower or "422" in error:
            return (
                f"Validation error for {tool_name}. "
                f"The provided data may not match GitHub's requirements. Details: {error}"
            )
        elif "rate limit" in error_lower or "429" in error:
            return (
                "GitHub API rate limit reached. "
                "Please wait before making more requests. Authenticated requests have higher limits. "
                "Consider reducing the frequency of operations."
            )
        elif "branch" in error_lower and "protected" in error_lower:
            return (
                f"Branch protection rules prevent this operation. "
                "The branch may require pull request reviews, status checks, or administrator privileges."
            )
        elif "invalid" in error_lower and "owner/repo" in error_lower:
            return (
                "Invalid repository format. Use 'owner/repo' format (e.g., 'facebook/react', not just 'react')."
            )
        else:
            return f"Error calling {tool_name}: {error}"
    
    async def cleanup(self):
        """Disconnect from GitHub with proper cleanup"""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass  # Suppress cleanup errors
        
        if self.stdio_context:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except Exception:
                pass  # Suppress cleanup errors
    
    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool to Gemini function declaration"""
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)
        
        if tool.inputSchema:
            schema = tool.inputSchema
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._clean_schema(prop_schema)
            
            if "required" in schema:
                parameters_schema.required.extend(schema["required"])
        
        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )
    
    def _clean_schema(self, schema: Dict) -> protos.Schema:
        """Convert JSON schema to protobuf schema"""
        schema_pb = protos.Schema()
        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(schema["type"], protos.Type.TYPE_UNSPECIFIED)
        if "description" in schema:
            schema_pb.description = schema["description"]
        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])
        if "items" in schema and isinstance(schema["items"], dict):
            schema_pb.items = self._clean_schema(schema["items"])
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._clean_schema(prop_schema)
        if "required" in schema:
            schema_pb.required.extend(schema["required"])
        return schema_pb
    
    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Convert protobuf types to Python types"""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value