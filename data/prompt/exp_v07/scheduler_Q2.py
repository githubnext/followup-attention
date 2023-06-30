# ****************************************************************************

# This helper allows you to perform some async operation and keep track of

# whether it's still ongoing, so in case someone wants to perform it again (and

# get a Promise for its completion) the same ongoing promise can be reused.


import asyncio
from typing import Awaitable, Callable, TypeVar, Generic, Union
from numpy import Inf

T = TypeVar('T')


class ScheduledTask(Generic[T]):
    def __init__(self,
            taskProducer: Callable[[], 'asyncio.Task[T]'],
            expirationMs: float = Inf):
        self.expirationMs = expirationMs
        self.taskProducer = taskProducer
        self._future: asyncio.Future[T] = None
        self._result: T = None

    async def run(self) -> T:
        print(self._future)
        if self._future is None:
            task = self.taskProducer()
            loop = asyncio.get_event_loop()
            self._future = loop.create_future()
            # run in background so this doesn't block
            asyncio.create_task(self._storeResult(task))
        return await self._future

    async def _storeResult(self, task: Awaitable[T]):
        try:
            self._result = await task
        finally:
            self._future.set_result(self._result)
            if (self._result is None):
                self._future = None # So this task can be tried again.
            if self.expirationMs < Inf and self._future is not None:
                await asyncio.sleep(self.expirationMs / 1000)
                self._future = None
                self._result = None
    def value(self) -> Union[T, None]:
        return self._result


# Question: What is the role of the parameter expirationMs?

# Answer: