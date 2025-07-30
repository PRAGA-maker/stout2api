import os
import psutil
import ray
import logging
import threading
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ScalingMode(Enum):
    SINGLE_REQUEST = "single_request"  # 100% resources for 1 request
    DUAL_REQUEST = "dual_request"      # 50% resources each for 2 requests  
    TRIPLE_REQUEST = "triple_request"  # 33% resources each for 3 requests

@dataclass
class SystemResources:
    total_cpus: int
    total_memory_gb: float
    available_cpus: int
    available_memory_gb: float
    cpu_usage_percent: float
    memory_usage_percent: float

@dataclass
class ActorConfig:
    num_cpus: float
    memory_gb: float
    max_concurrency: int
    scaling_mode: ScalingMode

class DynamicResourceManager:
    def __init__(self):
        self.system_resources = self._detect_system_resources()
        self.active_requests = 0
        self.request_lock = threading.Lock()
        self.last_scale_time = time.time()
        self.scale_cooldown = 1.0  # Minimum time between scaling decisions
        
        logger.info(f"Initialized DynamicResourceManager with {self.system_resources}")
        
    def _detect_system_resources(self) -> SystemResources:
        """Detect available system resources"""
        try:
            # Get CPU info
            total_cpus = psutil.cpu_count(logical=True)
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Get memory info
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent
            
            # Calculate available CPUs (reserve 1 CPU for system, but ensure at least 1 available)
            available_cpus = max(1, total_cpus - 1)
            
            # Ensure we have reasonable memory available (at least 1GB)
            available_memory_gb = max(1.0, available_memory_gb)
            
            resources = SystemResources(
                total_cpus=total_cpus,
                total_memory_gb=total_memory_gb,
                available_cpus=available_cpus,
                available_memory_gb=available_memory_gb,
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent
            )
            
            logger.info(f"Detected system resources: {resources}")
            return resources
            
        except Exception as e:
            logger.error(f"Failed to detect system resources: {e}")
            # Fallback to conservative defaults - will be detected dynamically
            return SystemResources(
                total_cpus=2,
                total_memory_gb=2.0,
                available_cpus=1,
                available_memory_gb=1.5,
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0
            )
    
    def _calculate_optimal_config(self, num_requests: int) -> ActorConfig:
        """Calculate optimal actor configuration based on number of active requests"""
        
        # Update system resources
        self.system_resources = self._detect_system_resources()
        
        # Cap the number of requests to prevent resource exhaustion
        effective_requests = min(num_requests, 3)
        
        # Calculate resource allocation percentages based on request count
        if effective_requests == 1:
            scaling_mode = ScalingMode.SINGLE_REQUEST
            # Use 100% of available resources for single request
            cpu_percentage = 1.0  # 100%
            memory_percentage = 0.8  # 80% (reserve 20% for system)
            max_concurrency = 1
            
        elif effective_requests == 2:
            scaling_mode = ScalingMode.DUAL_REQUEST
            # Use 50% of available resources per request
            cpu_percentage = 0.5  # 50%
            memory_percentage = 0.4  # 40% per request
            max_concurrency = 1
            
        else:  # effective_requests >= 3
            scaling_mode = ScalingMode.TRIPLE_REQUEST
            # Use 33% of available resources per request
            cpu_percentage = 0.33  # 33%
            memory_percentage = 0.25  # 25% per request
            max_concurrency = 1
        
        # Calculate actual resource allocation based on available system resources
        num_cpus = max(1.0, self.system_resources.available_cpus * cpu_percentage)
        memory_gb = max(1.0, self.system_resources.available_memory_gb * memory_percentage)
        
        # CRITICAL: Ensure we don't exceed available resources and add safety margins
        # Reserve more resources for system stability
        max_cpus = max(1.0, self.system_resources.available_cpus * 0.8)  # Reserve 20% for system
        max_memory = max(1.0, self.system_resources.available_memory_gb * 0.7)  # Reserve 30% for system
        
        num_cpus = min(num_cpus, max_cpus)
        memory_gb = min(memory_gb, max_memory)
        
        # Additional safety: never request more than 4 CPUs per actor to prevent resource exhaustion
        num_cpus = min(num_cpus, 4.0)
        
        config = ActorConfig(
            num_cpus=num_cpus,
            memory_gb=memory_gb,
            max_concurrency=max_concurrency,
            scaling_mode=scaling_mode
        )
        
        logger.info(f"Calculated config for {effective_requests} requests (capped from {num_requests}): {config}")
        return config
    
    def get_actor_config(self) -> ActorConfig:
        """Get current actor configuration based on active requests"""
        with self.request_lock:
            return self._calculate_optimal_config(self.active_requests)
    
    def register_request_start(self) -> ActorConfig:
        """Register that a new request is starting"""
        with self.request_lock:
            # Check if we're at the maximum concurrent requests limit
            if self.active_requests >= 3:  # MAX_CONCURRENT_REQUESTS
                logger.warning(f"Maximum concurrent requests (3) reached. Active: {self.active_requests}")
                # Still allow the request but log the warning
                
            self.active_requests += 1
            # CRITICAL: Cap active requests more aggressively to prevent runaway counting
            self.active_requests = min(self.active_requests, 5)  # Reduced from 10 to 5
            
            # Emergency reset if we have too many active requests
            if self.active_requests > 5:
                logger.error(f"EMERGENCY: Too many active requests ({self.active_requests}), resetting to 1")
                self.active_requests = 1
                
            config = self._calculate_optimal_config(self.active_requests)
            logger.info(f"Request started. Active requests: {self.active_requests}, Config: {config}")
            return config
    
    def register_request_end(self) -> ActorConfig:
        """Register that a request has ended"""
        with self.request_lock:
            self.active_requests = max(0, self.active_requests - 1)
            # Reset if we have an unreasonable number of active requests
            if self.active_requests > 5:  # Reduced from 10 to 5
                logger.warning(f"Resetting active requests from {self.active_requests} to 0")
                self.active_requests = 0
            config = self._calculate_optimal_config(self.active_requests)
            logger.info(f"Request ended. Active requests: {self.active_requests}, Config: {config}")
            return config
    
    def get_system_status(self) -> Dict:
        """Get current system status and resource usage"""
        self.system_resources = self._detect_system_resources()
        config = self.get_actor_config()
        
        return {
            "system_resources": {
                "total_cpus": self.system_resources.total_cpus,
                "total_memory_gb": round(self.system_resources.total_memory_gb, 2),
                "available_cpus": self.system_resources.available_cpus,
                "available_memory_gb": round(self.system_resources.available_memory_gb, 2),
                "cpu_usage_percent": round(self.system_resources.cpu_usage_percent, 2),
                "memory_usage_percent": round(self.system_resources.memory_usage_percent, 2)
            },
            "active_requests": self.active_requests,
            "current_config": {
                "num_cpus": config.num_cpus,
                "memory_gb": round(config.memory_gb, 2),
                "max_concurrency": config.max_concurrency,
                "scaling_mode": config.scaling_mode.value
            },
            "ray_resources": {
                "cluster_resources": ray.cluster_resources() if ray.is_initialized() else {},
                "available_resources": ray.available_resources() if ray.is_initialized() else {}
            }
        }
    
    def reset_request_count(self):
        """Reset the active request count (emergency recovery)"""
        with self.request_lock:
            old_count = self.active_requests
            self.active_requests = 0
            logger.warning(f"Reset active requests from {old_count} to 0")
            return self.active_requests

# Global resource manager instance
resource_manager = DynamicResourceManager() 