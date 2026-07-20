from __future__ import annotations

import asyncio
import logging
import time
import threading
from typing import Any


logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter for provider requests."""

    # 类变量用于存储单例实例
    _instances: dict[str, TokenBucket] = {}
    _lock = threading.Lock()

    def __new__(
        cls, bucket_id: str, capacity: int = 10, refill_rate: float = 1.0
    ) -> TokenBucket:
        """单例模式，确保相同bucket_id只有一个实例"""
        with cls._lock:
            if bucket_id not in cls._instances:
                instance = super(TokenBucket, cls).__new__(cls)
                setattr(instance, "_pending_args",  (bucket_id, capacity, refill_rate))
                cls._instances[bucket_id] = instance

            return cls._instances[bucket_id]

    def __init__(self, bucket_id: str, capacity: int = 10, refill_rate: float = 1.0):
        """初始化令牌桶

        Args:
            bucket_id: 令牌桶唯一标识符
            capacity: 令牌桶容量（最大令牌数）
            refill_rate: 令牌补充速率（令牌数/秒）
        """
        if getattr(self, "initialized", False):
            return


        pending_args = getattr(self, "_pending_args", None)

        if pending_args:
            bucket_id, capacity, refill_rate = pending_args

        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be greater than 0")

        self.bucket_id = bucket_id
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill_time = time.time()
        self._lock = threading.Lock()
        self.initialized = True

        logger.debug(
            "TokenBucket %s initialized: capacity=%s, refill_rate=%s",
            bucket_id,
            capacity,
            refill_rate,
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

        logger.debug(
            "TokenBucket %s refilled: added=%.2f, current=%.2f",
            self.bucket_id,
            tokens_to_add,
            self.tokens,
        )

    async def acquire(
        self, tokens_needed: int = 1, timeout: float | None = None
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
                    logger.debug(
                        "TokenBucket %s acquired %s tokens, remaining=%.2f",
                        self.bucket_id,
                        tokens_needed,
                        self.tokens,
                    )
                    return True

                # 计算等待时间：需要多久才能补充足够的令牌
                tokens_needed_to_wait = tokens_needed - self.tokens
                wait_time = tokens_needed_to_wait / self.refill_rate

            # 检查超时（在锁外检查）
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # 最多等待100ms，避免长时间阻塞
            wait_time = min(wait_time, 0.1)

            logger.debug(
                "TokenBucket %s waiting for refill: needed=%s, available=%.2f, wait=%.3fs",
                self.bucket_id,
                tokens_needed,
                self.tokens,
                wait_time,
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
                return True
            else:
                return False

    def get_available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill_tokens()
            return self.tokens

    def get_info(self) -> dict[str, Any]:
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
            logger.debug("TokenBucket %s reset, tokens=%s", self.bucket_id, self.tokens)

    def __repr__(self) -> str:
        """返回令牌桶的字符串表示"""
        return (
            f"TokenBucket(id={self.bucket_id}, capacity={self.capacity}, "
            f"refill_rate={self.refill_rate}, tokens={self.tokens:.2f})"
        )


class RateLimitManager:
    """速率限制管理器，管理多个令牌桶"""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
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

    def get_bucket(self, bucket_id: str) -> TokenBucket | None:
        """获取指定的令牌桶"""
        return self._buckets.get(bucket_id)

    def remove_bucket(self, bucket_id: str) -> bool:
        """移除指定的令牌桶"""
        with self._lock:
            if bucket_id in self._buckets:
                del self._buckets[bucket_id]
                return True
            return False

    def list_buckets(self) -> dict[str, dict[str, Any]]:
        """列出所有令牌桶的状态"""
        return {
            bucket_id: bucket.get_info() for bucket_id, bucket in self._buckets.items()
        }

    def reset_all(self) -> None:
        """重置所有令牌桶"""
        for bucket in self._buckets.values():
            bucket.reset()


rate_limit_manager = RateLimitManager()
