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

# === 1. ADDED COLORS ===
# We define ANSI escape codes for colors
class C:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
# ========================


# Configure Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Atlassian's official MCP Remote Proxy configuration
JIRA_SERVER_COMMAND = "npx"
JIRA_SERVER_ARGS = [
    "-y",
    "mcp-remote",
    "https://mcp.atlassian.com/v1/sse"
]

# The mcp-remote proxy handles auth via a browser
JIRA_ENV = {}


class JiraAgent:
    def __init__(self):
        self.session: ClientSession | None = None
        self.stdio_context = None
        self.model = genai.GenerativeModel('gemini-2.5-flash')
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
        
    async def connect_to_jira(self):
        """Connect to the Jira MCP server"""
        server_params = StdioServerParameters(
            command=JIRA_SERVER_COMMAND,
            args=JIRA_SERVER_ARGS,
            env={**os.environ, **JIRA_ENV}
        )
        
        self.stdio_context = stdio_client(server_params)
        self.stdio, self.write = await self.stdio_context.__aenter__()
        self.session = ClientSession(self.stdio, self.write)
        
        await self.session.__aenter__()
        await self.session.initialize()
        
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools
        
        # === UI CHANGE ===
        print(f"{C.GREEN}Connected to Jira MCP Server{C.ENDC}")
        print(f"{C.YELLOW}Available tools:{C.ENDC}")
        for tool in self.available_tools:
            print(f"{C.YELLOW}  - {tool.name}{C.ENDC}")
        # =================
        
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)
    
    def convert_mcp_tools_to_gemini(self) -> List[protos.FunctionDeclaration]:
        """Convert MCP tool definitions to Gemini protos.FunctionDeclaration objects"""
        gemini_tools = []
        
        for tool in self.available_tools:
            
            parameters_schema = protos.Schema(type_=protos.Type.OBJECT)

            if tool.inputSchema:
                schema = tool.inputSchema
                if "properties" in schema:
                    for prop_name, prop_schema in schema["properties"].items():
                        parameters_schema.properties[prop_name] = self._clean_schema(prop_schema)
                
                if "required" in schema:
                    parameters_schema.required.extend(schema["required"])

            function_declaration = protos.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=parameters_schema
            )
            
            gemini_tools.append(function_declaration)
        
        # === UI CHANGE (Colored debug output) ===
        print(f"{C.YELLOW}{'='*20} FINAL GEMINI TOOLS (PROTO) {'='*20}{C.ENDC}")
        try:
            for i, tool in enumerate(gemini_tools):
                print(f"{C.YELLOW}--- Tool {i+1}: {tool.name} ---{C.ENDC}")
                print(f"{C.YELLOW}{tool}{C.ENDC}")
        except Exception as e:
            print(f"{C.RED}Could not print tools: {e}{C.ENDC}")
        print(f"{C.YELLOW}{'='*60}{C.ENDC}")
        # =======================================
        
        return gemini_tools
    
    def _clean_schema(self, schema: Dict) -> protos.Schema:
        """
        Recursively convert a JSON schema dict to a protos.Schema object.
        """
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
        """Call an MCP tool and return the result"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        result = await self.session.call_tool(tool_name, arguments)
        return result

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """
        Recursively converts Protobuf composite types (MapComposite, RepeatedComposite)
        into standard Python dicts and lists.
        """
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
        if not self.session:
            await self.connect_to_jira()
        
        gemini_tools = self.convert_mcp_tools_to_gemini()
        
        if not self.chat:
            self.chat = self.model.start_chat(
                history=self.conversation_history
            )
        
        response = await self.chat.send_message_async(
            user_message,
            tools=gemini_tools
        )
        
        while response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            tool_name = function_call.name
            
            tool_args = self._deep_convert_proto_args(function_call.args)
            
            # === UI CHANGE (System/Tool call coloring) ===
            print(f"\n{C.MAGENTA}🔧 Calling tool: {C.BOLD}{tool_name}{C.ENDC}")
            print(f"{C.MAGENTA}   Arguments:{C.ENDC}")
            print(f"{C.YELLOW}{json.dumps(tool_args, indent=2)}{C.ENDC}")
            # ============================================
            
            try:
                tool_result = await self.call_mcp_tool(tool_name, tool_args)
                
                result_content = []
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        result_content.append(content.text)
                
                result_text = "\n".join(result_content)
                if not result_text:
                    result_text = json.dumps(tool_result.content, default=str)

                # === UI CHANGE ===
                print(f"{C.MAGENTA}   Result:{C.ENDC} {result_text[:200]}...")
                # =================
                
                response = await self.chat.send_message_async(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"result": result_text}
                            )
                        )]
                    ),
                    tools=gemini_tools
                )
                
            except Exception as e:
                error_msg = f"Error calling tool {tool_name}: {str(e)}"
                # === UI CHANGE ===
                print(f"   {C.RED}❌ {error_msg}{C.ENDC}")
                # =================
                
                response = await self.chat.send_message_async(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"error": error_msg}
                            )
                        )]
                    ),
                    tools=gemini_tools
                )
        
        self.conversation_history = self.chat.history
        return response.text
    
    async def run_interactive(self):
        """Run an interactive chat session"""
        # === UI CHANGE (Main chat loop coloring) ===
        print("=" * 60)
        print(f"{C.BOLD}🤖 Jira AI Agent (Powered by {self.model.model_name}){C.ENDC}")
        print("=" * 60)
        print(f"{C.YELLOW}\nConnecting to Jira...{C.ENDC}")
        
        try:
            await self.connect_to_jira()
        except Exception:
            print(f"{C.RED}❌ FAILED TO CONNECT TO JIRA.{C.ENDC}")
            traceback.print_exc()
            return

        print(f"\n{C.GREEN}✅ Connected!{C.ENDC} You can now chat with your Jira assistant.")
        print("Type 'exit' or 'quit' to end the conversation.\n")
        
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
        # ==========================================


async def main():
    """Main entry point"""
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        # === UI CHANGE ===
        print(f"{C.RED}❌ Missing required environment variables:{C.ENDC}")
        for var in missing_vars:
            print(f"{C.RED}   - {var}{C.ENDC}")
        print(f"{C.YELLOW}\nPlease set these variables before running the agent.{C.ENDC}")
        # =================
        return
    
    agent = JiraAgent()
    await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())