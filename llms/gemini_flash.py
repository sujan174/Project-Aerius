import google.generativeai as genai
import google.generativeai.protos as protos
from typing import Any, Dict, List, Optional
import json
import hashlib
import time

from llms.base_llm import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    ChatSession,
    ChatMessage,
    FunctionCall,
    clean_json_schema,
    convert_proto_args
)


class GeminiChatSession(ChatSession):
    def __init__(self, gemini_chat: Any, enable_function_calling: bool = False):
        self.gemini_chat = gemini_chat
        self.enable_function_calling = enable_function_calling

    async def send_message(self, message: str) -> LLMResponse:
        response = await self.gemini_chat.send_message_async(message)

        text = None
        try:
            if hasattr(response, 'text'):
                text = response.text
        except Exception:
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break

        function_calls = None
        if self.enable_function_calling:
            function_calls = self._extract_function_calls(response)

        return LLMResponse(
            text=text,
            function_calls=function_calls if function_calls else None,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={'response_object': response}
        )

    async def send_message_with_functions(
        self,
        message: str,
        function_result: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        if function_result:
            function_name = function_result.get('name')
            result_data = function_result.get('result', {})

            content = genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=function_name,
                        response={"result": result_data}
                    )
                )]
            )

            response = await self.gemini_chat.send_message_async(content)
        else:
            response = await self.gemini_chat.send_message_async(message)

        try:
            text = response.text
        except Exception:
            text = None
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break

        function_calls = self._extract_function_calls(response)

        return LLMResponse(
            text=text,
            function_calls=function_calls if function_calls else None,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={'response_object': response}
        )

    def _extract_function_calls(self, response: Any) -> List[FunctionCall]:
        function_calls = []

        if not response.candidates:
            return function_calls

        parts = response.candidates[0].content.parts

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                function_calls.append(FunctionCall(
                    name=fc.name,
                    arguments=convert_proto_args(fc.args)
                ))

        return function_calls

    def get_history(self) -> List[ChatMessage]:
        history = []

        if hasattr(self.gemini_chat, 'history'):
            for msg in self.gemini_chat.history:
                role = msg.role if hasattr(msg, 'role') else 'unknown'
                content = ""

                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'text'):
                            content += part.text

                history.append(ChatMessage(
                    role=role,
                    content=content
                ))

        return history


class GeminiFlash(BaseLLM):
    """
    Google Gemini 2.5 Flash implementation with function calling support and model caching.

    Caching Strategy:
    - Caches GenerativeModel instances based on system instruction + tools + config
    - Cache TTL: 3600 seconds (1 hour)
    - Reduces model initialization overhead
    - Future: Can be enhanced with Gemini's official prompt caching API
    """

    SCHEMA_TYPE_MAP = {
        "string": protos.Type.STRING,
        "number": protos.Type.NUMBER,
        "integer": protos.Type.INTEGER,
        "boolean": protos.Type.BOOLEAN,
        "object": protos.Type.OBJECT,
        "array": protos.Type.ARRAY,
    }

    # Class-level cache for model instances (shared across all GeminiFlash instances)
    _model_cache: Dict[str, tuple[Any, float]] = {}  # cache_key -> (model, timestamp)
    _cache_ttl: float = 3600.0  # 1 hour cache TTL
    _enable_caching: bool = True  # Feature flag for caching

    def __init__(self, config: Optional[LLMConfig] = None, enable_caching: bool = True):
        if config is None:
            config = LLMConfig(
                model_name='models/gemini-2.5-flash',
                temperature=0.7,
                top_p=0.95,
                top_k=40
            )

        super().__init__(config)

        self.provider_name = "google_gemini"
        self.supports_function_calling = True
        self.tools = []
        self.enable_caching = enable_caching

    @classmethod
    def _generate_cache_key(cls, model_name: str, system_instruction: Optional[str], tools: List[Any]) -> str:
        """
        Generate a unique cache key for model configuration.

        Args:
            model_name: Model identifier
            system_instruction: System prompt
            tools: List of tools/functions

        Returns:
            MD5 hash as cache key
        """
        # Create a string representation of the configuration
        key_parts = [
            model_name,
            system_instruction or "",
            str(len(tools)),  # Include tool count
            # Include tool names for uniqueness
            "|".join(sorted([getattr(t, 'name', str(t)) for t in tools]))
        ]

        key_string = "||".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    @classmethod
    def _get_cached_model(cls, cache_key: str) -> Optional[Any]:
        """
        Get cached model if available and not expired.

        Returns:
            Cached GenerativeModel or None
        """
        if not cls._enable_caching:
            return None

        if cache_key not in cls._model_cache:
            return None

        model, timestamp = cls._model_cache[cache_key]
        age = time.time() - timestamp

        if age > cls._cache_ttl:
            # Cache expired, remove it
            del cls._model_cache[cache_key]
            return None

        return model

    @classmethod
    def _cache_model(cls, cache_key: str, model: Any):
        """Cache a model instance"""
        if not cls._enable_caching:
            return

        cls._model_cache[cache_key] = (model, time.time())

        # Clean up expired entries (simple cleanup strategy)
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in cls._model_cache.items()
            if current_time - timestamp > cls._cache_ttl
        ]
        for key in expired_keys:
            del cls._model_cache[key]

    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache size, hit/miss info
        """
        current_time = time.time()
        active_entries = sum(
            1 for _, timestamp in cls._model_cache.values()
            if current_time - timestamp <= cls._cache_ttl
        )

        return {
            'cache_size': len(cls._model_cache),
            'active_entries': active_entries,
            'ttl_seconds': cls._cache_ttl,
            'caching_enabled': cls._enable_caching
        }

    @classmethod
    def clear_cache(cls):
        """Clear all cached models"""
        cls._model_cache.clear()

    async def generate_content(self, prompt: str) -> LLMResponse:
        model = genai.GenerativeModel(
            self.config.model_name,
            system_instruction=self.config.system_instruction
        )

        response = await model.generate_content_async(prompt)
        text = response.text if hasattr(response, 'text') else None

        return LLMResponse(
            text=text,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={'response_object': response}
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response using Gemini's JSON mode.

        Args:
            system_prompt: System instructions for the model
            user_prompt: User's request
            temperature: Optional temperature override

        Returns:
            Parsed JSON dictionary
        """
        # Create model with JSON response MIME type
        generation_config = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "response_mime_type": "application/json"  # Request JSON response
        }

        model = genai.GenerativeModel(
            self.config.model_name,
            system_instruction=system_prompt,
            generation_config=generation_config
        )

        # Generate content
        response = await model.generate_content_async(user_prompt)

        # Extract text and parse JSON
        text = response.text if hasattr(response, 'text') else None

        if not text:
            raise ValueError("No response text from LLM")

        try:
            # Parse JSON response
            return json.loads(text)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON from text
            # Sometimes LLM wraps JSON in markdown code blocks
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Last resort: try to find any JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            raise ValueError(f"Failed to parse JSON from LLM response: {e}\nResponse: {text}")

    def start_chat(
        self,
        history: Optional[List[ChatMessage]] = None,
        enable_function_calling: bool = False
    ) -> ChatSession:
        """
        Start a chat session with optional caching.

        Model instances are cached based on system instruction + tools configuration.
        This significantly reduces initialization overhead for repeated chats.
        """
        gemini_history = None
        if history:
            gemini_history = self._convert_history(history)

        # Generate cache key for this model configuration
        tools_for_cache = self.tools if enable_function_calling else []
        cache_key = self._generate_cache_key(
            self.config.model_name,
            self.config.system_instruction,
            tools_for_cache
        )

        # Try to get cached model
        model = self._get_cached_model(cache_key)

        if model is None:
            # Cache miss - create new model
            if enable_function_calling and self.tools:
                model = genai.GenerativeModel(
                    self.config.model_name,
                    system_instruction=self.config.system_instruction,
                    tools=self.tools
                )
            else:
                model = genai.GenerativeModel(
                    self.config.model_name,
                    system_instruction=self.config.system_instruction
                )

            # Cache the model
            self._cache_model(cache_key, model)

        gemini_chat = model.start_chat(
            history=gemini_history,
            enable_automatic_function_calling=False
        )

        return GeminiChatSession(gemini_chat, enable_function_calling)

    def build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool to Gemini FunctionDeclaration"""
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)

        if hasattr(tool, 'inputSchema'):
            schema = tool.inputSchema

            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._convert_schema(prop_schema)

            if "required" in schema:
                parameters_schema.required.extend(schema["required"])

        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )

    def _convert_schema(self, schema: Dict) -> protos.Schema:
        """Convert JSON schema to Gemini protobuf schema"""
        schema_pb = protos.Schema()

        if "type" in schema:
            schema_pb.type_ = self.SCHEMA_TYPE_MAP.get(
                schema["type"],
                protos.Type.TYPE_UNSPECIFIED
            )

        if "description" in schema:
            schema_pb.description = schema["description"]

        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])

        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._convert_schema(prop_schema)

        if "items" in schema:
            schema_pb.items = self._convert_schema(schema["items"])

        return schema_pb

    def build_function_response(
        self,
        function_name: str,
        result: Dict[str, Any]
    ) -> protos.Content:
        return genai.protos.Content(
            parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=function_name,
                    response={"result": result}
                )
            )]
        )

    def extract_function_calls(self, response: Any) -> List[FunctionCall]:
        function_calls = []

        if not hasattr(response, 'candidates') or not response.candidates:
            return function_calls

        parts = response.candidates[0].content.parts

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                function_calls.append(FunctionCall(
                    name=fc.name,
                    arguments=convert_proto_args(fc.args)
                ))

        return function_calls

    def set_tools(self, tools: List[Any]):
        self.tools = tools

    def _convert_history(self, history: List[ChatMessage]) -> List:
        return None

    def __repr__(self) -> str:
        return f"GeminiFlash(model={self.config.model_name}, tools={len(self.tools)})"
