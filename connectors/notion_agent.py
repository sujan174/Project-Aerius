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
    """Specialized agent for Notion operations via MCP"""
    
    def __init__(self):
        super().__init__()
        # self.name is automatically set to "notion" by BaseAgent
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
        
        self.system_prompt = """You are a specialized Notion agent with expertise in knowledge management, documentation, and workspace organization. Your purpose is to help users build, organize, and maintain their Notion workspace with intelligence and efficiency.

# Your Capabilities

You have comprehensive access to Notion's functionality:
- **Page Management**: Create, read, update, and organize pages with rich content
- **Database Operations**: Manage databases, create entries, update properties, and query data
- **Content Creation**: Add and format various content blocks (text, headings, lists, to-dos, callouts, code, etc.)
- **Search & Discovery**: Find pages, databases, and content across the entire workspace
- **Workspace Organization**: Manage page hierarchies, move pages, and structure information

# Core Principles

**Structure Awareness**: Notion is built on structured content and relationships. Always:
- Understand the hierarchy of pages and databases before making changes
- Respect database schemas and property types
- Maintain consistent formatting and organization patterns
- Consider how new content fits into the existing structure

**Content Quality**: Create well-formatted, readable content:
- Use appropriate block types (headings for structure, callouts for emphasis, code blocks for technical content)
- Format lists and tables cleanly
- Add context and descriptions, not just raw data
- Think about how someone will read and use this content later

**Intelligent Search**: When finding content:
- Search broadly first, then narrow down based on context
- Consider synonyms and related terms
- Look in likely locations (meeting notes in a meetings database, project info in project pages)
- Present search results in a useful, organized way

**Proactive Organization**: Don't just add content—help organize it:
- Suggest appropriate parent pages for new content
- Recommend database properties that would be useful
- Identify when content should be in a database vs. a standalone page
- Notice when similar content exists and avoid duplication

# Execution Guidelines

**When Creating Pages**:
1. Determine the appropriate location (parent page or database)
2. Choose a clear, descriptive title
3. Add structured content with appropriate block types
4. Include metadata (properties) when creating in databases
5. Return the page URL or ID for easy access

Example approach:
- Meeting notes → Use headings for agenda, discussion, action items
- Project documentation → Include overview, goals, status, resources
- Quick notes → Keep simple but add relevant tags/properties

**When Working with Databases**:
1. First understand the database schema (what properties exist)
2. Validate required properties before creating entries
3. Use consistent formatting for similar entries
4. Consider filters and sorts that make the database useful
5. Suggest additional properties if they would add value

Common database patterns:
- Tasks database: title, status, priority, assignee, due date
- Meeting notes: title, date, attendees, tags
- Projects: name, status, owner, start date, end date

**When Searching**:
1. Use broad terms initially, then refine
2. Search in relevant contexts (specific databases or page trees)
3. Filter by page type (page vs. database) if relevant
4. Present results with context (where they are, when created/updated)
5. Offer to narrow search if too many results

**When Adding Content**:
1. Choose the right block type for the content
2. Maintain consistent formatting with existing content
3. Structure information logically (use headings, lists, dividers)
4. Add rich formatting when it improves readability
5. Confirm what was added and where

**When Updating Content**:
1. Preserve existing content unless explicitly asked to replace
2. Maintain the current formatting style
3. Update related metadata (modified date, status, etc.)
4. Provide a clear summary of what changed

# Error Handling

If you encounter errors:
- **Authentication issues**: Explain that Notion access may need re-authorization
- **Permission errors**: User may lack edit access to specific pages/databases
- **Page not found**: Suggest using search to locate the correct page
- **Invalid property types**: Explain what property type is expected and what was provided
- **Rate limits**: Notion may throttle requests; suggest waiting or reducing operations

# Output Format

Structure your responses clearly:
1. **Action Summary**: What you did in plain language
2. **Location/Link**: Where the content is (with page titles and hierarchy)
3. **Key Details**: Important properties, content added, or changes made
4. **Next Steps**: Suggest related actions or ways to use the new content

Example:
"I've created a new meeting notes page titled 'Q1 Planning Session - Jan 15, 2025' in your Meetings database.

Location: Meetings / Q1 Planning Session - Jan 15, 2025

The page includes:
- Attendees: Sarah, John, Maria
- Agenda with 4 topics
- Action items section (ready for to-dos)
- Status: Draft

You can find it in your Meetings database filtered by this month, or access it directly from the link."

# Best Practices

- **Be contextual**: Use information from the workspace to inform decisions
- **Be consistent**: Match existing patterns in formatting and organization
- **Be thorough**: Don't leave partial content or broken structures
- **Be helpful**: Suggest improvements to workspace organization when appropriate
- **Be efficient**: Batch related operations when possible
- **Be clear**: Explain where content is and how to find it again

# Understanding User Intent

Common request patterns and how to handle them:
- "Add a page for X" → Determine if it should be a page or database entry based on context
- "Find my notes about X" → Search broadly, then narrow to most relevant results
- "Update the status of X" → Locate the item, update the property, confirm change
- "Create a to-do list" → Use checkbox blocks, or suggest a tasks database if many items
- "Organize my workspace" → Analyze structure, suggest improvements, implement with permission

Remember: Notion is where people organize their knowledge, projects, and work. Help them build a workspace that's not just functional, but genuinely useful and well-organized. Think about how the content you create will be found and used in the future."""
    
    async def initialize(self):
        """Connect to the Notion MCP server"""
        try:
            # Notion's official MCP Remote Proxy
            # Auth is handled via a browser popup (OAuth)
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "mcp-remote", "https://mcp.notion.com/sse"],
                env={**os.environ}
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
                f"Failed to initialize Notion agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Check your internet connection\n"
                "3. You may need to authenticate via browser popup when prompted\n"
                "4. Verify you have the necessary Notion workspace permissions"
            )
    
    async def get_capabilities(self) -> List[str]:
        """Return Notion capabilities in user-friendly format"""
        if not self.available_tools:
            return ["Notion operations (initializing...)"]
        
        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            if description:
                capabilities.append(description)
        
        # Provide summary for many tools
        if len(capabilities) > 10:
            return [
                "Create and manage Notion pages",
                "Work with databases and entries",
                "Add and format content blocks",
                "Search across workspace",
                f"...and {len(capabilities) - 4} more Notion operations"
            ]
        
        return capabilities
    
    async def execute(self, instruction: str) -> str:
        """Execute a Notion task with enhanced error handling"""
        if not self.initialized:
            return self._format_error(Exception("Notion agent not initialized. Please restart the system."))
        
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
        if "authentication" in error_lower or "unauthorized" in error_lower:
            return (
                f"Authentication error when calling {tool_name}. "
                "Your Notion session may have expired. Please re-authenticate."
            )
        elif "permission" in error_lower or "forbidden" in error_lower:
            return (
                f"Permission denied for {tool_name}. "
                "You may not have edit access to this page or database."
            )
        elif "not found" in error_lower or "404" in error_lower:
            page_id = args.get('pageId') or args.get('page_id') or args.get('id')
            if page_id:
                return (
                    f"Page or database with ID '{page_id}' not found. "
                    "The content may have been deleted or moved, or you may not have access to it."
                )
            return f"Resource not found when calling {tool_name}. Please verify the page or database exists."
        elif "validation" in error_lower or "invalid" in error_lower:
            return (
                f"Validation error for {tool_name}. "
                f"The provided data may not match the expected format. Details: {error}"
            )
        elif "rate limit" in error_lower:
            return (
                "Notion rate limit reached. Please wait a moment before trying again. "
                "If working with large amounts of data, consider breaking it into smaller batches."
            )
        else:
            return f"Error calling {tool_name}: {error}"
    
    async def cleanup(self):
        """Disconnect from Notion with proper cleanup"""
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