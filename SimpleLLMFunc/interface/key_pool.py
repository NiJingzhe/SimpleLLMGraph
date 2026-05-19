import heapq
from typing import List, Tuple, Dict
from SimpleLLMFunc.logger import push_critical, get_location
import threading # 导入 threading 模块

class APIKeyPool:
    # 类变量用于存储单例实例
    _instances: Dict[str, 'APIKeyPool'] = {}

    def __new__(cls, api_keys: List[str], provider_id: str) -> 'APIKeyPool':
        # 如果已经为这个 app_id 创建了实例，返回现有实例
        if provider_id in cls._instances:
            return cls._instances[provider_id]

        # 创建新实例
        instance = super(APIKeyPool, cls).__new__(cls)
        cls._instances[provider_id] = instance
        return instance

    def __init__(self, api_keys: List[str], provider_id: str) -> None:
        # 如果已经初始化，跳过初始化过程
        if hasattr(self, 'initialized') and self.initialized:   # type: ignore
            return

        if len(api_keys) == 0 or api_keys is None:
            push_critical(
                f"API key pool {provider_id} is empty. Please check your configuration.",
                location=get_location()
            )

            raise ValueError(
                f"API key pool {provider_id} is empty. Please check your configuration."
            )


        self.api_keys = api_keys
        self.app_id = provider_id

        # 内存中的存储，替代 Redis
        self.heap: List[Tuple[float, str]] = [(0, key) for key in self.api_keys]
        heapq.heapify(self.heap)
        self.key_to_task_count: Dict[str, int] = {key: 0 for key in self.api_keys}
        # 维护 key 到堆中索引的映射，用于 O(1) 查找
        self.key_to_index: Dict[str, int] = {key: i for i, (_, key) in enumerate(self.heap)}

        self.lock = threading.Lock() # 为每个实例创建一个锁
        self.initialized = True

    def get_least_loaded_key(self) -> str:
        with self.lock: # 获取锁保护读操作
            # 获取任务数量最小的 API key
            if not self.heap:
                raise ValueError(f"{self.app_id} has no available API keys")
            return self.heap[0][1]

    def increment_task_count(self, api_key: str) -> None:
        with self.lock: # 获取锁
            if api_key not in self.key_to_task_count:
                raise ValueError(f"API key {api_key} is not in the pool")

            # 增加任务计数
            self.key_to_task_count[api_key] += 1

            # 更新堆
            self._update_heap(api_key, self.key_to_task_count[api_key])

    def decrement_task_count(self, api_key: str) -> None:
        with self.lock: # 获取锁
            if api_key not in self.key_to_task_count:
                raise ValueError(f"API key {api_key} is not in the pool")

            # 减少任务计数
            self.key_to_task_count[api_key] -= 1

            # 更新堆
            self._update_heap(api_key, self.key_to_task_count[api_key])

    def _update_heap(self, api_key: str, new_task_count: int) -> None:
        # 使用映射快速找到元素在堆中的位置 - O(1)
        if api_key not in self.key_to_index:
            raise ValueError(f"API key {api_key} is not in the heap")

        index = self.key_to_index[api_key]
        old_count, _ = self.heap[index]

        # 更新堆中对应位置的元素
        self.heap[index] = (new_task_count, api_key)

        # 根据新值与旧值的关系，决定调整方向
        # 如果新值更小，需要向上冒泡（向堆顶）
        # 如果新值更大，需要向下沉（向堆底）
        if new_task_count < old_count:
            # 新值更小，向上冒泡到堆顶方向 - O(log n)
            self._siftdown_with_index_update(0, index)
        elif new_task_count > old_count:
            # 新值更大，向下沉到堆底方向 - O(log n)
            self._siftup_with_index_update(index)
        # 如果 new_task_count == old_count，不需要调整

    def _siftup_with_index_update(self, pos: int) -> None:
        """向下沉调整（向堆底）并更新受影响元素的索引映射 - O(log n)

        当元素值增大时，需要向下沉到合适位置
        """
        heap = self.heap
        endpos = len(heap)
        newitem = heap[pos]

        # 向下沉：找到最小的子节点，如果新值大于子节点，则交换
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            # 选择较小的子节点
            if rightpos < endpos and heap[childpos][0] > heap[rightpos][0]:
                childpos = rightpos
            # 如果新值小于等于最小子节点，停止下沉
            if newitem[0] <= heap[childpos][0]:
                break
            # 移动子节点到父节点位置
            heap[pos] = heap[childpos]
            # 更新被移动元素的索引
            self.key_to_index[heap[childpos][1]] = pos
            pos = childpos
            childpos = 2 * pos + 1

        # 放置新元素到最终位置
        heap[pos] = newitem
        # 更新新元素的索引
        self.key_to_index[newitem[1]] = pos

    def _siftdown_with_index_update(self, startpos: int, pos: int) -> None:
        """向上冒泡调整（向堆顶）并更新受影响元素的索引映射 - O(log n)

        当元素值减小时，需要向上冒泡到合适位置
        """
        heap = self.heap
        newitem = heap[pos]

        # 向上冒泡：与父节点比较，如果新值小于父节点，则交换
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            # 如果新值大于等于父节点，停止冒泡
            if newitem[0] >= parent[0]:
                break
            # 移动父节点到子节点位置
            heap[pos] = parent
            # 更新被移动元素的索引
            self.key_to_index[parent[1]] = pos
            pos = parentpos

        # 放置新元素到最终位置
        heap[pos] = newitem
        # 更新新元素的索引
        self.key_to_index[newitem[1]] = pos
