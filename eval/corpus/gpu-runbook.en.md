# GPU Training Runbook

Owner: Platform Reliability — Last reviewed 2026-02-18

## CUDA out of memory during training

Symptom: the job aborts with `RuntimeError: CUDA out of memory. Tried to allocate 2.20 GiB`.

The most common cause is a batch size that grew after a config change, not a leak. Check the effective batch size first: gradient accumulation multiplies the per-device batch. Reduce `per_device_train_batch_size` to 8 and set `gradient_accumulation_steps` to 4 to keep the effective batch at 32.

If memory is still exhausted, enable gradient checkpointing. It trades roughly 30 percent extra step time for a 40 percent reduction in activation memory.

## Fragmented memory

When `nvidia-smi` reports free memory but allocation still fails, the allocator is fragmented. Set the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before starting the process. Restarting the worker clears fragmentation permanently only if the dataloader is not holding pinned buffers.

## NCCL timeout on multi-node runs

Symptom: `Watchdog caught collective operation timeout` after exactly 1800 seconds.

Raise `NCCL_TIMEOUT` only after confirming that no rank is stuck on a slow data shard. A rank that reads from cold object storage can stall the entire all-reduce. The default watchdog window is 30 minutes.

## Checkpoint corruption

If a checkpoint fails to load with an unexpected EOF, the write was interrupted. Always write to a temporary path and rename atomically. Retention policy keeps the last five checkpoints and one checkpoint per epoch for ninety days.

## Escalation

Page the on-call platform engineer if a production training job fails three consecutive times. The escalation channel is #ml-platform-oncall and the response target is fifteen minutes during business hours.
