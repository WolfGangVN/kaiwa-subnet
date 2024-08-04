import asyncio

from communex.module.module import Module, endpoint
from loguru import logger

from starlette.responses import StreamingResponse, JSONResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from kaiwa_subnet.base.config import KaiwaBaseSettings


class InferenceEngineV2(Module):
    def __init__(self, settings: KaiwaBaseSettings) -> None:
        super().__init__()
        engine_args = AsyncEngineArgs(
            enforce_eager=True,
            model=settings.model,
            tokenizer=settings.model,
            dtype="half",
            max_model_len=2048,
            quantization="gptq",
            gpu_memory_utilization=settings.gpu_memory_utilization,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        model_config = asyncio.run(self.engine.get_model_config())
        served_model_names = [engine_args.model]
        response_role = "assistant"
        lora_modules = None
        chat_template = asyncio.run(self.engine.get_tokenizer()).chat_template
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            served_model_names,
            response_role,
            lora_modules,
            chat_template,
        )

    @endpoint
    async def chat(self, input: dict, timeout: int = 120) -> dict:
        if not input.messages or not input.messages[0]["content"]:
            return ErrorResponse(
                error="Invalid input: messages must be a list with at least one non-empty message"
            )
        try:        
            import aiohttp
            async with aiohttp.ClientSession(timeout=timeout) as session:
                return await session.post(
                    "http://localhost:8000/infer",
                    json={"req": input.messages[0]["content"]},
                )
        except Exception as e:
            return ErrorResponse(error=str(e))
        # resp = await self.openai_serving_chat.create_chat_completion(
        #     ChatCompletionRequest.model_validate(input)
        # )
        # data = resp.model_dump()
        # logger.debug(data)
        # return data

    @endpoint
    def get_metadata(self) -> dict:
        return {"models": self.models}


if __name__ == "__main__":
    model_name = "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit"
    d = InferenceEngine(settings=KaiwaBaseSettings(model=model_name))
    out = asyncio.run(
        d.chat(
            input=ChatCompletionRequest(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the remainder when 732^732 is divided by 27?",
                    }
                ],
            )
        )
    )
    print(out)
