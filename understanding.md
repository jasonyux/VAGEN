## rollout


```python
class RayPPOTrainer:
    def _process_in_mini_batches(self,batch, rollout_manager, mini_batch_size):
        """
        Process the batch in mini-batches.
        
        Args:
            batch: DataProto containing the data
            rollout_manager: Manager for rollout operations
            mini_batch_size: Size of each mini-batch to process
        
        Returns:
            Tuple of (final_combined_batch_output, combined_rst)
        """
        batch_size = len(batch)
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size  # Ceiling division
        
        all_final_gen_batch_outputs = []
        all_rst = []
        
        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, batch_size)
            actual_mini_batch_size = end_idx - start_idx
            print(f"Processing mini-batch {i+1}/{num_mini_batches}, size: {actual_mini_batch_size}")
            
            # Extract env_configs for this mini-batch
            mini_batch_env_configs = [
                batch.non_tensor_batch['extra_info'][j]
                for j in range(start_idx, end_idx)
            ]
            
            ####### MAIN ROLLOUT CODE #######
            rollout_manager.reset(mini_batch_env_configs) # prepare environment
            rollout_manager.rollout_loop() # gen rollouts
            mini_batch_output = rollout_manager.generate_batch_for_update() # convert rollout to trainable data
            mini_batch_rst = rollout_manager.recording_to_log() # collect stats for wandb logging
            ####### MAIN ROLLOUT CODE END #######

            # Store results
            all_final_gen_batch_outputs.append(mini_batch_output)
            all_rst.extend(mini_batch_rst)  # Extend the list since rst is a list
        
       
            combined_output = DataProto.concat(all_final_gen_batch_outputs)
       
        
        return combined_output, all_rst
```


## Environment


The main entry point for environments to be used during rollouts is:
- via env APIs `vagen.rollout.qwen_rollout.rollout_manager.QwenVLRolloutManager`
- via service APIs `vagen.rollout.qwen_rollout.rollout_manager_service.QwenVLRolloutManagerService`

These classes provide code to 1) create the env class; 2) rollout with vllm engine; 3) pipe data pack to trainer.


### Env config data format

The configurations used for creating env class via `env.reset()` are stored in the dataset parquet file.
Each task instance is prepared as a row in the parquet file, an example being:
```json
{
    "data_source": "sokoban",  # needed
    "prompt": [  # placeholder
        {
            "content": "",
            "role": "user"
        }
    ],
    "extra_info": {   # used to create env class and call reset() before rollouts
        "env_config": {  # used by REGISTERED_ENV[env_name]['config_cls'](**env_config)
            "render_mode": "vision"
        },
        "env_name": "sokoban",
        "seed": 1,  # used by env.reset(seed)
        "split": "train" # not used meta info
    }
}
```