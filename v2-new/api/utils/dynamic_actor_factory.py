import ray
import logging
import gc
from typing import Optional, Dict, Any
from api.utils.resource_manager import resource_manager, ActorConfig
from api.models import TranslationType

logger = logging.getLogger(__name__)

class DynamicActorFactory:
    def __init__(self):
        self.actors = {}  # Cache of created actors
        self.actor_lock = None  # Will be initialized when Ray is available
        
    def _get_actor_key(self, config: ActorConfig) -> str:
        """Generate a unique key for actor configuration"""
        return f"actor_{config.num_cpus}_{config.memory_gb}_{config.max_concurrency}"
    
    def _create_actor(self, config: ActorConfig):
        """Create a new Ray actor with the specified configuration"""
        
        @ray.remote(
            num_cpus=config.num_cpus,
            memory=config.memory_gb * 1024 * 1024 * 1024,  # Convert GB to bytes
            max_restarts=5,  # Increased from 2
            max_task_retries=5,  # Increased from 2
            max_concurrency=config.max_concurrency
        )
        class DynamicInferenceActor:
            def __init__(self):
                # Suppress TensorFlow warnings about duplicate registrations
                import os
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
                
                from api.stout_service import stout_service
                self.stout_service = stout_service
                self.config = config
                
            def infer(self, input_text, translation_type):
                try:
                    if translation_type == TranslationType.FORWARD:
                        result = self.stout_service.translate_forward(input_text)
                    else:
                        result = self.stout_service.translate_reverse(input_text)
                    gc.collect()
                    return result
                except Exception as e:
                    logger.error(f"Actor inference failed: {e}")
                    raise
                    
            def get_config(self):
                return {
                    "num_cpus": self.config.num_cpus,
                    "memory_gb": self.config.memory_gb,
                    "max_concurrency": self.config.max_concurrency,
                    "scaling_mode": self.config.scaling_mode.value
                }
        
        # Create the actor
        actor = DynamicInferenceActor.remote()
        logger.info(f"Created new actor with config: {config}")
        return actor
    
    def get_optimal_actor(self) -> Any:
        """Get the optimal actor for current load"""
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized")
            
        # Get current optimal configuration
        config = resource_manager.get_actor_config()
        actor_key = self._get_actor_key(config)
        
        # Check if we have a cached actor with this configuration
        if actor_key in self.actors:
            try:
                # Test if the actor is still alive
                ray.get(self.actors[actor_key].get_config.remote(), timeout=1.0)
                logger.debug(f"Using cached actor with config: {config}")
                return self.actors[actor_key]
            except Exception as e:
                logger.warning(f"Cached actor failed health check: {e}")
                # Remove failed actor from cache
                del self.actors[actor_key]
        
        # Create new actor with optimal configuration
        try:
            new_actor = self._create_actor(config)
            
            # Test the new actor with shorter timeout
            test_config = ray.get(new_actor.get_config.remote(), timeout=1.5)  # Reduced from 2.0
            logger.info(f"New actor created and tested successfully: {test_config}")
            
            # Cache the actor
            self.actors[actor_key] = new_actor
            
            # Clean up old actors if we have too many cached
            if len(self.actors) > 3:  # Increased from 2 to 3
                self._cleanup_old_actors()
            
            return new_actor
            
        except Exception as e:
            logger.error(f"Failed to create actor with config {config}: {e}")
            # Fallback to a basic actor
            return self._create_fallback_actor()
    
    def _create_fallback_actor(self):
        """Create a fallback actor with minimal resources"""
        from api.utils.resource_manager import ScalingMode
        
        # Get current system resources for fallback calculation
        system_resources = resource_manager.system_resources
        
        # Use minimal resources: 1 CPU and 20% of available memory (min 1GB, max 2GB)
        fallback_cpus = 1.0
        fallback_memory = max(1.0, min(2.0, system_resources.available_memory_gb * 0.2))
        
        fallback_config = ActorConfig(
            num_cpus=fallback_cpus,
            memory_gb=fallback_memory,
            max_concurrency=1,
            scaling_mode=ScalingMode.SINGLE_REQUEST
        )
        
        logger.warning(f"Creating fallback actor with minimal resources: {fallback_config}")
        
        # Clean up all existing actors before creating fallback
        self.cleanup_all_actors()
        
        return self._create_actor(fallback_config)
    
    def _cleanup_old_actors(self):
        """Clean up old actors from cache"""
        if len(self.actors) <= 3:  # Increased from 2 to 3
            return
            
        # Remove oldest actors (simple FIFO for now)
        actor_keys = list(self.actors.keys())
        for old_key in actor_keys[:-3]:  # Keep only the 3 most recent
            try:
                # Gracefully terminate the actor
                ray.kill(self.actors[old_key])
                del self.actors[old_key]
                logger.info(f"Cleaned up old actor: {old_key}")
            except Exception as e:
                logger.warning(f"Failed to cleanup actor {old_key}: {e}")
                # Remove from cache anyway
                if old_key in self.actors:
                    del self.actors[old_key]
    
    def get_actor_status(self) -> Dict[str, Any]:
        """Get status of all cached actors"""
        status = {
            "cached_actors": len(self.actors),
            "actor_configs": {}
        }
        
        for key, actor in self.actors.items():
            try:
                config = ray.get(actor.get_config.remote(), timeout=1.0)
                status["actor_configs"][key] = config
            except Exception as e:
                status["actor_configs"][key] = {"error": str(e)}
        
        return status
    
    def cleanup_all_actors(self):
        """Clean up all cached actors"""
        for key, actor in self.actors.items():
            try:
                ray.kill(actor)
                logger.info(f"Cleaned up actor: {key}")
            except Exception as e:
                logger.warning(f"Failed to cleanup actor {key}: {e}")
        
        self.actors.clear()
    
    def force_resource_cleanup(self):
        """Force cleanup when resources are exhausted"""
        logger.warning("Forcing resource cleanup due to exhaustion")
        self.cleanup_all_actors()
        
        # Force Ray to release resources
        try:
            import ray
            if ray.is_initialized():
                # Trigger garbage collection
                import gc
                gc.collect()
                logger.info("Forced resource cleanup completed")
        except Exception as e:
            logger.error(f"Failed to force resource cleanup: {e}")

# Global actor factory instance
actor_factory = DynamicActorFactory() 