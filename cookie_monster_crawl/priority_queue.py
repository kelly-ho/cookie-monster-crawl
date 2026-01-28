import asyncio
import heapq
import random

class AsyncPriorityQueue(asyncio.Queue):
    def _init(self, maxsize):
        self._queue = []

    def _put(self, item):
        priority, payload = item
        tie_breaker = random.random()
        heapq.heappush(self._queue, (priority, tie_breaker, payload))

    def _get(self):
        priority, _, payload = heapq.heappop(self._queue)
        return priority, payload
