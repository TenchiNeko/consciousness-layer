# Consciousness Layer for Standalone Orchestrator

An experimental add-on that gives autonomous coding agents emotional memory, behavioral learning, and intrinsic motivation.

## What This Does

- **Behavioral Weights** - learns from experience (procedural memory, not memorization)
- **Split-Brain** - 80B experiencer + 7B integrator with separate roles
- **Book Rewards** - earns reading time through good work, cross-graded by peer model
- **Activity Preferences** - discovers what it enjoys by comparing emotions across activities
- **LoRA Training** - fine-tune personality adapters from accumulated experience

## Install

```bash
cd ~/standalone-orchestrator
git clone https://github.com/TenchiNeko/consciousness-layer.git
bash consciousness-layer/deploy-v31-full.sh
```

## Monitor

```bash
watch -n 60 python3 monitor.py
```

## Structure

- modules/ - 13 consciousness modules
- patches/ - 4 orchestrator patches (idempotent)
- tools/ - training data export + LoRA fine-tuning
- books/ - reading reward content

## License

MIT
