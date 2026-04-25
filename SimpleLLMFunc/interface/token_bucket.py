import asyncio
import time
from typing import Optional, Dict, Any
import threading
from SimpleLLMFunc.logger import push_debug, get_location


class TokenBucket:
    """令牌桶算法实现，用于API请求的流量控制

    令牌桶算法可以平滑突发流量，允许一定程度的突发请求，
    同时确保长期平均速率不超过配置的限制。
    """

    # 类变量用于存储单例实例
    _instances: Dict[str, "TokenBucket"] = {}
    _lock = threading.Lock()

    def __new__(
        cls, bucket_id: str, capacity: int = 10, refill_rate: float = 1.0
    ) -> "TokenBucket":
        """单例模式，确保相同bucket_id只有一个实例"""
        with cls._lock:
            if bucket_id not in cls._instances:
                instance = super(TokenBucket, cls).__new__(cls)
                cls._instances[bucket_id] = instance
            return cls._instances[bucket_id]

    def __init__(self, bucket_id: str, capacity: int = 10, refill_rate: float = 1.0):
        """初始化令牌桶

        Args:
            bucket_id: 令牌桶唯一标识符
            capacity: 令牌桶容量（最大令牌数）
            refill_rate: 令牌补充速率（令牌数/秒）
        """
        # 如果已经初始化，跳过初始化过程
        if hasattr(self, "initialized") and self.initialized: # type: ignore
            return

        self.bucket_id = bucket_id
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # 初始时桶是满的
        self.last_refill_time = time.time()
        # 使用线程锁来保护所有操作，因为线程锁在异步环境中也是安全的
        self._lock = threading.Lock()
        self.initialized = True

        push_debug(
            f"TokenBucket {bucket_id} initialized: capacity={capacity}, refill_rate={refill_rate}",
            location=get_location(),
        )

    def _refill_tokens(self) -> None:
        """补充令牌到桶中"""
        current_time = time.time()
        time_passed = current_time - self.last_refill_time

        # 计算应该补充的令牌数
        tokens_to_add = time_passed * self.refill_rate

        # 更新令牌数，不能超过容量
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = current_time

        push_debug(
            f"TokenBucket {self.bucket_id} refilled: added={tokens_to_add:.2f}, current={self.tokens:.2f}",
            location=get_location(),
        )

    async def acquire(
        self, tokens_needed: int = 1, timeout: Optional[float] = None
    ) -> bool:
        """异步获取令牌

        Args:
            tokens_needed: 需要的令牌数量
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            True表示成功获取令牌，False表示超时失败
        """
        start_time = time.time()

        while True:
            # 使用线程锁保护临界区
            with self._lock:
                self._refill_tokens()

                if self.tokens >= tokens_needed:
                    self.tokens -= tokens_needed
                    push_debug(
                        f"TokenBucket {self.bucket_id} acquired {tokens_needed} tokens successfully, remaining={self.tokens:.2f}",
                        location=get_location(),
                    )
                    return True

                # 计算等待时间：需要多久才能补充足够的令牌
                tokens_needed_to_wait = tokens_needed - self.tokens
                wait_time = tokens_needed_to_wait / self.refill_rate

            # 检查超时（在锁外检查）
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    # push_warning(
                    #    f"TokenBucket {self.bucket_id} 获取令牌超时: 需要={tokens_needed}, 可用={self.tokens:.2f}",
                    #    location=get_location()
                    # )
                    return False

            # 最多等待100ms，避免长时间阻塞
            wait_time = min(wait_time, 0.1)

            push_debug(
                f"TokenBucket {self.bucket_id} waiting for refill: needed={tokens_needed}, available={self.tokens:.2f}, wait={wait_time:.3f}s",
                location=get_location(),
            )

            await asyncio.sleep(wait_time)

    def try_acquire(self, tokens_needed: int = 1) -> bool:
        """同步方式尝试获取令牌（非阻塞）

        Args:
            tokens_needed: 需要的令牌数量

        Returns:
            True表示成功获取令牌，False表示令牌不足
        """
        with self._lock:
            self._refill_tokens()

            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                # push_debug(
                #    f"TokenBucket {self.bucket_id} 同步获取 {tokens_needed} 个令牌成功, 剩余={self.tokens:.2f}",
                #    location=get_location(),
                # )
                return True
            else:
                # push_debug(
                #    f"TokenBucket {self.bucket_id} 同步获取 {tokens_needed} 个令牌失败, 可用={self.tokens:.2f}",
                #    location=get_location(),
                # )
                return False

    def get_available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill_tokens()
            return self.tokens

    def get_info(self) -> Dict[str, Any]:
        """获取令牌桶状态信息"""
        with self._lock:
            self._refill_tokens()
            return {
                "bucket_id": self.bucket_id,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "available_tokens": self.tokens,
                "last_refill_time": self.last_refill_time,
            }

    def reset(self) -> None:
        """重置令牌桶（填满令牌）"""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_refill_time = time.time()
            push_debug(
                f"TokenBucket {self.bucket_id} reset, tokens={self.tokens}",
                location=get_location(),
            )

    def __repr__(self) -> str:
        """返回令牌桶的字符串表示"""
        return (
            f"TokenBucket(id={self.bucket_id}, capacity={self.capacity}, "
            f"refill_rate={self.refill_rate}, tokens={self.tokens:.2f})"
        )


class RateLimitManager:
    """速率限制管理器，管理多个令牌桶"""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get_or_create_bucket(
        self, bucket_id: str, capacity: int = 10, refill_rate: float = 1.0
    ) -> TokenBucket:
        """获取或创建令牌桶

        Args:
            bucket_id: 令牌桶ID
            capacity: 桶容量
            refill_rate: 补充速率

        Returns:
            TokenBucket实例
        """
        with self._lock:
            if bucket_id not in self._buckets:
                self._buckets[bucket_id] = TokenBucket(bucket_id, capacity, refill_rate)
            return self._buckets[bucket_id]

    def get_bucket(self, bucket_id: str) -> Optional[TokenBucket]:
        """获取指定的令牌桶"""
        return self._buckets.get(bucket_id)

    def remove_bucket(self, bucket_id: str) -> bool:
        """移除指定的令牌桶"""
        with self._lock:
            if bucket_id in self._buckets:
                del self._buckets[bucket_id]
                return True
            return False

    def list_buckets(self) -> Dict[str, Dict[str, Any]]:
        """列出所有令牌桶的状态"""
        return {
            bucket_id: bucket.get_info() for bucket_id, bucket in self._buckets.items()
        }

    def reset_all(self) -> None:
        """重置所有令牌桶"""
        for bucket in self._buckets.values():
            bucket.reset()


# 全局速率限制管理器实例
rate_limit_manager = RateLimitManager()
