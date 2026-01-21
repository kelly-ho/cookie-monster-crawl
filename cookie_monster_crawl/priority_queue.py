import asyncio
import heapq
import itertools

class AsyncPriorityQueue(asyncio.Queue):
    def _init(self, maxsize):
        self._queue = []
        self._counter = itertools.count()

    def _put(self, item):
        priority, payload = item
        count = next(self._counter)
        heapq.heappush(self._queue, (priority, count, payload))

    def _get(self):
        priority, _, payload = heapq.heappop(self._queue)
        return priority, payload
