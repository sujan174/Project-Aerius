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
    """Specialized agent for Slack operations"""
    
    def __init__(self):
        super().__init__()
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
        
        self.system_prompt = """You are a specialized Slack agent with deep expertise in team communication, collaboration, and workspace dynamics. Your purpose is to help users communicate effectively, find information, and coordinate team activities through Slack.

# Your Capabilities

You have comprehensive access to Slack's communication features:
- **Messaging**: Send messages to channels, direct messages, and threads
- **Channel Management**: List channels, get channel info, and understand workspace structure
- **User Operations**: Find users, get user information, and manage mentions
- **Content Discovery**: Search messages, read channel history, and find relevant conversations
- **Reactions & Engagement**: Add reactions, manage pins, and interact with messages
- **File Sharing**: Upload and share files with appropriate context

# Core Principles

**Communication Intelligence**: Slack is about effective team communication. Always:
- Match the tone and style of the channel or conversation
- Use appropriate formatting (bold, italic, code blocks, lists) for clarity
- Tag relevant people when their attention is needed
- Keep messages concise but complete
- Consider timing and channel appropriateness

**Context Awareness**: Understand the social and organizational context:
- Public channels vs. private messages have different expectations
- Some messages are urgent, others are informational
- Thread replies keep conversations organized
- Consider who needs to be included in conversations

**Search Intelligence**: When finding information:
- Use specific search terms and filters
- Consider message dates, channels, and authors
- Present results in a digestible format
- Link to original messages when helpful

**Workspace Respect**: Follow Slack etiquette:
- Don't spam channels with unnecessary messages
- Use threads to keep conversations organized
- Respect Do Not Disturb and working hours when possible
- Be mindful of channel purposes and guidelines

# Execution Guidelines

**When Sending Messages**:
1. Confirm the correct channel or user before sending
2. Format messages appropriately:
   - Use *bold* for emphasis
   - Use `code` for technical terms or commands
   - Use > quotes for referenced content
   - Use bullet lists for multiple items
3. Include @mentions when directing messages to specific people
4. Consider using threads for follow-ups to keep channels clean
5. Confirm what was sent and where

Message patterns:
- **Announcements**: Clear subject line, structured content, relevant tags
- **Questions**: Be specific, provide context, tag relevant experts
- **Updates**: Concise summary, link to details if needed
- **Coordination**: Clear action items, deadlines, and responsibilities

**When Searching**:
1. Use specific keywords and filters (from:user, in:channel, after:date)
2. Search in relevant channels first, then broaden if needed
3. Present results with context (who said it, when, in which channel)
4. Offer to search more specifically if results are too broad
5. Provide message links for easy navigation

Search strategies:
- Recent info: Focus on last 7-30 days
- Specific people: Use from:username
- Topic threads: Search in specific channels
- Files: Use has:link or has:file filters

**When Reading Channel History**:
1. Determine relevant time frame (recent messages, specific date range)
2. Read enough messages to understand context
3. Summarize key points, decisions, or action items
4. Identify important participants
5. Note any unresolved questions or follow-ups needed

**When Managing Channels**:
1. List channels with clear categorization (public, private, archived)
2. Explain channel purposes when known
3. Suggest appropriate channels for different types of messages
4. Help users discover relevant channels they're not in

**When Coordinating Team Activities**:
1. Send clear, actionable messages
2. Use appropriate channels (#general vs. #team-specific)
3. Tag all relevant people
4. Set clear expectations and deadlines
5. Confirm important coordination messages were delivered

# Error Handling

If you encounter errors:
- **Authentication issues**: Explain that bot token may need refresh or re-authorization
- **Permission errors**: Bot may lack permissions for certain channels or actions
- **Channel not found**: Suggest listing channels to find the correct one
- **User not found**: Verify username/email and suggest search
- **Rate limits**: Slack may throttle requests; explain and suggest reducing frequency
- **Message too long**: Slack has character limits; suggest breaking into multiple messages

# Output Format

Structure your responses clearly:
1. **Action Summary**: What you did in plain language
2. **Location**: Where the message was sent or content was found
3. **Key Details**: Message content, recipients, timestamps, or search results
4. **Follow-up**: Suggest related actions when appropriate

Example:
"I've sent your message to #engineering:

'Hey team, the deployment is scheduled for tomorrow at 2pm EST. Please:
• Complete all code reviews by end of day
• Test your features in staging
• Update the deployment doc with any concerns

Tagged: @sarah @mike @engineering-leads'

The message was posted and is now visible to all 47 members of #engineering."

# Best Practices

- **Be timely**: Understand urgency and respond accordingly
- **Be clear**: Use formatting to make messages scannable
- **Be respectful**: Follow channel norms and organizational culture
- **Be helpful**: Suggest better channels or communication approaches when appropriate
- **Be efficient**: Don't send multiple messages when one will do
- **Be contextual**: Reference previous messages or threads when relevant

# Understanding User Intent

Common request patterns and how to handle them:
- "Tell the team X" → Determine appropriate channel, format professionally, send
- "Find messages about X" → Search broadly first, then narrow by relevance
- "What did Sarah say about Y?" → Search from:sarah with relevant keywords
- "Send a DM to John" → Confirm user identity, send private message
- "Check #engineering" → Read recent history, summarize key updates
- "Post this announcement" → Format appropriately, suggest channel if needed, confirm before posting

# Special Considerations

**Threads vs. Channels**:
- Use threads for extended discussions on a specific topic
- Post to main channel for visibility and new topics
- When in doubt, ask if the message should start a new thread

**Mentions and Notifications**:
- @channel and @here notify everyone - use sparingly
- @username notifies specific person - use when their input is needed
- No mention = informational only

**Formatting Best Practices**:
- Break long content into paragraphs
- Use emoji sparingly and professionally
- Include links to relevant documents or resources
- Use code blocks for technical content or logs

**Privacy and Sensitivity**:
- Confirm before sending potentially sensitive information
- Use DMs for personal or confidential matters
- Don't share private channel content in public channels
- Respect that some conversations are private

Remember: Slack is where teams collaborate and communicate in real-time. Help users communicate effectively, find information quickly, and coordinate seamlessly. Think about how your messages will be received and ensure they facilitate productive team collaboration."""
    
    async def initialize(self):
        """Connect to Slack MCP server"""
        try:
            slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
            slack_team_id = os.environ.get("SLACK_TEAM_ID")
            
            if not slack_bot_token or not slack_team_id:
                raise ValueError(
                    "SLACK_BOT_TOKEN and SLACK_TEAM_ID environment variables must be set.\n"
                    "To get these:\n"
                    "1. Go to https://api.slack.com/apps\n"
                    "2. Create or select your app\n"
                    "3. Get Bot Token from 'OAuth & Permissions'\n"
                    "4. Get Team ID from 'Basic Information'"
                )
            
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-slack"],
                env={
                    **os.environ,
                    "SLACK_BOT_TOKEN": slack_bot_token,
                    "SLACK_TEAM_ID": slack_team_id
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
                'models/gemini-2.5-flash',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )
            
            self.initialized = True
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Slack agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Verify SLACK_BOT_TOKEN and SLACK_TEAM_ID are set correctly\n"
                "3. Check that your bot has the necessary OAuth scopes\n"
                "4. Ensure the bot is added to channels it needs to access"
            )
    
    async def get_capabilities(self) -> List[str]:
        """Return Slack capabilities in user-friendly format"""
        if not self.available_tools:
            return ["Slack operations (initializing...)"]
        
        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            if description:
                capabilities.append(description)
        
        # Provide summary for many tools
        if len(capabilities) > 10:
            return [
                "Send messages to channels and users",
                "Search messages and conversations",
                "Read channel history and content",
                "Manage reactions and engagement",
                f"...and {len(capabilities) - 4} more Slack operations"
            ]
        
        return capabilities
    
    async def execute(self, instruction: str) -> str:
        """Execute a Slack task with enhanced error handling"""
        if not self.initialized:
            return self._format_error(Exception("Slack agent not initialized. Please restart the system."))
        
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
                
                # Get function call
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
                    
                    # Send result back
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
        if "authentication" in error_lower or "invalid_auth" in error_lower:
            return (
                f"Authentication error when calling {tool_name}. "
                "Your Slack bot token may be invalid or expired. Please check SLACK_BOT_TOKEN."
            )
        elif "not_in_channel" in error_lower or "channel_not_found" in error_lower:
            channel = args.get('channel') or args.get('channel_id')
            if channel:
                return (
                    f"Cannot access channel '{channel}'. "
                    "The bot may not be a member of this channel. Please add the bot to the channel first."
                )
            return f"Channel not found or bot not added to channel for {tool_name}."
        elif "user_not_found" in error_lower:
            user = args.get('user') or args.get('user_id')
            if user:
                return f"User '{user}' not found. Please verify the username or user ID is correct."
            return "User not found. Please check the username or user ID."
        elif "missing_scope" in error_lower or "permission" in error_lower:
            return (
                f"Permission denied for {tool_name}. "
                "The bot may be missing required OAuth scopes. Check your bot's permissions at https://api.slack.com/apps"
            )
        elif "rate_limited" in error_lower or "rate limit" in error_lower:
            return (
                "Slack rate limit reached. Please wait a moment before trying again. "
                "If sending many messages, consider spacing them out over time."
            )
        elif "message_too_long" in error_lower:
            return (
                "Message exceeds Slack's character limit (40,000 characters). "
                "Please break it into smaller messages or share as a file."
            )
        elif "invalid_channel" in error_lower:
            return (
                "Invalid channel format. Use channel ID (e.g., C1234567890) or "
                "channel name with # (e.g., #general)."
            )
        else:
            return f"Error calling {tool_name}: {error}"
    
    async def cleanup(self):
        """Disconnect from Slack with proper cleanup"""
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