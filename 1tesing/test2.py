import os
import json
import asyncio
import traceback
from typing import Any, Dict, List
import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

# === 1. ANSI COLOR CODES ===
class C:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# === 2. CONFIGURE ALL SERVERS ===

# Atlassian's official MCP Remote Proxy configuration
JIRA_SERVER_CONFIG = {
    "command": "npx",
    "args": ["-y", "mcp-remote", "https://mcp.atlassian.com/v1/sse"],
    "env": {}  # mcp-remote handles auth via browser
}

# Official @modelcontextprotocol/server-slack
SLACK_SERVER_CONFIG = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-slack"],
    "env": {
        "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", ""),  # xoxb-...
        "SLACK_TEAM_ID": os.environ.get("SLACK_TEAM_ID", "")       # T...
    }
}

# Configure Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)


async def test_slack_connection():
    """Test if Slack credentials are valid"""
    try:
        import aiohttp
    except ImportError:
        print(f"{C.YELLOW}⚠ aiohttp not installed. Skipping Slack validation. Install with: pip install aiohttp{C.ENDC}")
        return True
    
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    team_id = os.environ.get("SLACK_TEAM_ID")
    
    if not bot_token or not team_id:
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {bot_token}"}
            async with session.get("https://slack.com/api/auth.test", headers=headers) as resp:
                data = await resp.json()
                if data.get("ok"):
                    print(f"{C.GREEN}✓ Slack credentials valid! Team: {data.get('team')}{C.ENDC}")
                    return True
                else:
                    print(f"{C.RED}✗ Slack auth failed: {data.get('error')}{C.ENDC}")
                    return False
    except Exception as e:
        print(f"{C.RED}✗ Could not test Slack connection: {e}{C.ENDC}")
        return False


class TerminalAgent:
    def __init__(self, system_prompt=""):
        self.sessions: Dict[str, ClientSession] = {}
        self.stdio_contexts = []
        self.tool_to_server_map: Dict[str, str] = {}
        
        # Define all servers to connect to
        self.servers_config = {
            "jira": JIRA_SERVER_CONFIG,
            "slack": SLACK_SERVER_CONFIG
        }

        self.system_prompt = system_prompt
        self.model = None  # Will be created after tools are loaded
        
        self.chat = None
        self.conversation_history = []
        self.available_tools = []
        
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }
    
    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        
        print(f"{C.YELLOW}Connecting to MCP servers...{C.ENDC}")
        
        for server_name, config in self.servers_config.items():
            print(f"{C.YELLOW}  - Connecting to {server_name}...{C.ENDC}")
            try:
                server_params = StdioServerParameters(
                    command=config["command"],
                    args=config["args"],
                    env={**os.environ, **config["env"]}
                )
                
                context = stdio_client(server_params)
                stdio, write = await context.__aenter__()
                session = ClientSession(stdio, write)
                
                await session.__aenter__()
                await session.initialize()
                
                # Store the session and context
                self.sessions[server_name] = session
                self.stdio_contexts.append(context)
                
                print(f"{C.GREEN}  ✓ Connected to {server_name}!{C.ENDC}")
                
            except Exception as e:
                print(f"{C.RED}  ✗ FAILED to connect to {server_name}: {e}{C.ENDC}")
                traceback.print_exc()
        
    async def disconnect(self):
        """Disconnect from all MCP servers"""
        print(f"\n{C.YELLOW}Disconnecting from servers...{C.ENDC}")
        for session in self.sessions.values():
            if session:
                try:
                    await session.__aexit__(None, None, None)
                except:
                    pass
        for context in self.stdio_contexts:
            if context:
                try:
                    await context.__aexit__(None, None, None)
                except:
                    pass
        
    async def convert_mcp_tools_to_gemini(self) -> List[protos.FunctionDeclaration]:
        """Convert MCP tool definitions from all servers to Gemini format"""
        
        print(f"{C.YELLOW}{'='*20} LOADING ALL TOOLS {'='*20}{C.ENDC}")
        
        gemini_tools = []
        self.tool_to_server_map.clear()
        self.available_tools.clear()

        for server_name, session in self.sessions.items():
            try:
                tools_list = await session.list_tools()
                print(f"{C.YELLOW}--- Tools from {server_name}: ---{C.ENDC}")
                
                for tool in tools_list.tools:
                    self.available_tools.append(tool)
                    
                    # Convert tool to Gemini proto
                    function_declaration = self._build_function_declaration(tool)
                    gemini_tools.append(function_declaration)
                    
                    # Map the tool name to its server
                    self.tool_to_server_map[tool.name] = server_name
                    print(f"{C.GREEN}  ✓ Loaded {tool.name}{C.ENDC}")

            except Exception as e:
                print(f"{C.RED}  ✗ Error loading tools from {server_name}: {e}{C.ENDC}")
                traceback.print_exc()

        print(f"{C.YELLOW}{'='*50}{C.ENDC}")
        print(f"{C.GREEN}Total tools loaded: {len(gemini_tools)}{C.ENDC}\n")
        return gemini_tools

    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Helper to convert a single MCP tool to a proto"""
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
        """Recursively convert a JSON schema dict to a protos.Schema object."""
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
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool on the correct server"""
        
        # Find which server this tool belongs to
        server_name = self.tool_to_server_map.get(tool_name)
        
        if not server_name:
            raise RuntimeError(f"Unknown tool '{tool_name}'. Not found in any server.")
            
        # Get the correct session
        session = self.sessions.get(server_name)
        
        if not session:
            raise RuntimeError(f"No active session found for server '{server_name}'.")

        print(f"{C.MAGENTA}   (Routing to {C.BOLD}{server_name}{C.ENDC}{C.MAGENTA} server...){C.ENDC}")
        result = await session.call_tool(tool_name, arguments)
        return result

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively converts Protobuf composite types into standard Python dicts/lists."""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {
                k: self._deep_convert_proto_args(v) 
                for k, v in value.items()
            }
        elif "RepeatedComposite" in type_str:
            return [
                self._deep_convert_proto_args(item) 
                for item in value
            ]
        else:
            return value

    async def process_message(self, user_message: str) -> str:
        """Process a user message and return the response"""
        
        # Connect and get tools *once* at the start of the first message
        if not self.chat:
            await self.connect_to_servers()
            
            if not self.sessions:
                raise RuntimeError("Failed to connect to any MCP servers!")
            
            gemini_tools = await self.convert_mcp_tools_to_gemini()
            
            if not gemini_tools:
                raise RuntimeError("No tools were loaded from any server!")
            
            # Create model with tools
            # Use gemini-2.5-pro for better compatibility with function calling
            self.model = genai.GenerativeModel(
                'models/gemini-2.5-flash',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )
            
            self.chat = self.model.start_chat(
                history=self.conversation_history
            )
        
        response = await self.chat.send_message_async(user_message)
        
        # Handle function calling loop
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # Check if there's a function call in the response
            parts = response.candidates[0].content.parts
            has_function_call = any(part.function_call for part in parts if hasattr(part, 'function_call'))
            
            if not has_function_call:
                break
                
            # Get the first function call
            function_call = None
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    break
            
            if not function_call:
                break
                
            tool_name = function_call.name
            
            tool_args = self._deep_convert_proto_args(function_call.args)
            
            print(f"\n{C.MAGENTA}🔧 Calling tool: {C.BOLD}{tool_name}{C.ENDC}")
            print(f"{C.MAGENTA}   Arguments:{C.ENDC}")
            print(f"{C.YELLOW}{json.dumps(tool_args, indent=2)}{C.ENDC}")
            
            try:
                tool_result = await self.call_mcp_tool(tool_name, tool_args)
                
                result_content = []
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        result_content.append(content.text)
                
                result_text = "\n".join(result_content)
                if not result_text:
                    result_text = json.dumps(tool_result.content, default=str)

                print(f"{C.MAGENTA}   Result:{C.ENDC} {result_text[:200]}...")
                
                response = await self.chat.send_message_async(
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
                error_msg = f"Error calling tool {tool_name}: {str(e)}"
                print(f"   {C.RED}❌ {error_msg}{C.ENDC}")
                
                response = await self.chat.send_message_async(
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
            print(f"{C.YELLOW}⚠ Warning: Reached maximum function call iterations{C.ENDC}")
        
        self.conversation_history = self.chat.history
        return response.text
    
    async def run_interactive(self):
        """Run an interactive chat session"""
        print("=" * 60)
        print(f"{C.BOLD}🤖 Multi-Tool AI Agent (Gemini 1.5 Pro){C.ENDC}")
        print("=" * 60)
        
        try:
            while True:
                user_input = input(f"{C.BOLD}{C.CYAN}You: {C.ENDC}").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print(f"\n{C.GREEN}Goodbye! 👋{C.ENDC}")
                    break
                
                if not user_input:
                    continue
                
                try:
                    response = await self.process_message(user_input)
                    print(f"\n{C.BOLD}{C.GREEN}Assistant:{C.ENDC}\n{response}\n")
                    
                except Exception as e:
                    print(f"\n{C.RED}❌ An error occurred: {str(e)}{C.ENDC}")
                    print("="*20, "FULL TRACEBACK", "="*20)
                    traceback.print_exc()
                    print("="*58 + "\n")
        
        finally:
            await self.disconnect()


async def main():
    """Main entry point"""
    required_vars = {
        "GOOGLE_API_KEY": "Google API Key is missing.",
        "SLACK_BOT_TOKEN": "SLACK_BOT_TOKEN is missing (for Slack server).",
        "SLACK_TEAM_ID": "SLACK_TEAM_ID is missing (for Slack server)."
    }
    
    missing_vars = [msg for var, msg in required_vars.items() if not os.environ.get(var)]
    
    if missing_vars:
        print(f"{C.RED}❌ Missing required environment variables:{C.ENDC}")
        for msg in missing_vars:
            print(f"{C.RED}   - {msg}{C.ENDC}")
        print(f"\n{C.YELLOW}Please set these variables in your .env file before running.{C.ENDC}")
        print(f"{C.YELLOW}Setup Instructions:{C.ENDC}")
        print(f"{C.YELLOW}1. SLACK_BOT_TOKEN: Create a Slack app at https://api.slack.com/apps{C.ENDC}")
        print(f"{C.YELLOW}   - Add these scopes: channels:history, channels:read, chat:write, reactions:write, users:read{C.ENDC}")
        print(f"{C.YELLOW}   - Install to workspace and copy the Bot User OAuth Token (xoxb-...){C.ENDC}")
        print(f"{C.YELLOW}2. SLACK_TEAM_ID: Find it in your Slack URL (https://app.slack.com/client/T0123ABCDE/...){C.ENDC}")
        return
    
    # Test Slack connection before starting
    print(f"{C.YELLOW}Testing Slack connection...{C.ENDC}")
    if not await test_slack_connection():
        print(f"{C.RED}⚠ Slack authentication test failed. The agent will try to connect anyway.{C.ENDC}")
        print(f"{C.YELLOW}If you encounter issues, verify your SLACK_BOT_TOKEN and SLACK_TEAM_ID.{C.ENDC}\n")
    
    prompt = """
    You are a helpful and efficient assistant with access to Jira and Slack.
    - You will be friendly, concise, and professional.
    - When asked to do something, you will use your available tools.
    - For Jira tasks, you will always confirm the issue key (e.g., "PROJ-123").
    - For Slack, you will confirm the channel name or ID before sending a message.
    - When listing Slack channels, present them in a clear, readable format.
    """
    
    agent = TerminalAgent(system_prompt=prompt)
    await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())