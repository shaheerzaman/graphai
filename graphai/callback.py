import asyncio
from typing import Optional
from collections.abc import AsyncIterator
from semantic_router.utils.logger import logger


log_stream = True

class Callback:
    first_token = True
    current_node_name: Optional[str] = None
    active: bool = True
    queue: asyncio.Queue

    def __init__(self):
        self.queue = asyncio.Queue()

    def __call__(self, token: str, node_name: Optional[str] = None):
        self._check_node_name(node_name=node_name)
        # otherwise we just assume node is correct and send token
        self.queue.put_nowait(token)
    
    async def acall(self, token: str, node_name: Optional[str] = None):
        self._check_node_name(node_name=node_name)
        # otherwise we just assume node is correct and send token
        self.queue.put_nowait(token)
    
    async def aiter(self) -> AsyncIterator[str]:
        """Used by receiver to get the tokens from the stream queue. Creates
        a generator that yields tokens from the queue until the END token is
        received.
        """
        while True:
            token = await self.queue.get()
            yield token
            self.queue.task_done()
            if token == "<graphai:END>":
                break

    async def start_node(self, node_name: str, active: bool = True):
        self.current_node_name = node_name
        if self.first_token:
            # TODO JB: not sure if we need self.first_token
            self.first_token = False
        self.active = active
        if self.active:
            self.queue.put_nowait(f"<graphai:start:{node_name}>")
    
    async def end_node(self, node_name: str):
        self.current_node_name = None
        if self.active:
            self.queue.put_nowait(f"<graphai:end:{node_name}>")

    async def close(self):
        self.queue.put_nowait("<graphai:END>")

    def _check_node_name(self, node_name: Optional[str] = None):
        if node_name:
            # we confirm this is the current node
            if self.current_node_name != node_name:
                raise ValueError(
                    f"Node name mismatch: {self.current_node_name} != {node_name}"
                )