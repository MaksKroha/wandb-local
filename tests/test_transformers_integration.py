"""End-to-end: transformers' Trainer with report_to="wandb" (skipped without torch/transformers)."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class _Data(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        ids = torch.tensor([(i + j) % 50 for j in range(8)])
        return {"input_ids": ids, "labels": ids}


def test_trainer_reports_to_wandb_local(tmp_path, monkeypatch):
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
    from transformers.integrations import WandbCallback, is_wandb_available

    assert is_wandb_available()
    monkeypatch.setenv("WANDB_PROJECT", "hf-test")
    monkeypatch.setenv("WANDB_LOG_MODEL", "checkpoint")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=16, n_embd=16, n_layer=1, n_head=2))
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"), report_to="wandb", run_name="hf-run", max_steps=6, logging_steps=2,
        save_steps=3, eval_strategy="steps", eval_steps=3, per_device_train_batch_size=4, use_cpu=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=_Data(), eval_dataset=_Data())
    assert any(isinstance(cb, WandbCallback) for cb in trainer.callback_handler.callbacks)
    trainer.train()
    import wandb

    run = wandb.run
    assert run is not None and run.name == "hf-run" and run.project == "hf-test"
    wandb.finish()
    rd = Path(run.run_dir)
    rows = [json.loads(l) for l in (rd / "files" / "wandb-history.jsonl").read_text().splitlines()]
    keys = set().union(*(r.keys() for r in rows))
    assert {"train/loss", "train/global_step", "eval/loss"} <= keys
    summary = json.loads((rd / "files" / "wandb-summary.json").read_text())
    assert "train_runtime" in summary and "train/global_step" in summary
    info = json.loads((rd / "wandb-run.json").read_text())
    assert info["labels"]["code"] == "transformers_trainer"
    assert any(m["name"] == "*" and m["step_metric"] == "train/global_step" for m in info["metrics"])
    aliases = {a for art in info["artifacts"] for a in art["aliases"]}
    assert "checkpoint_global_step_3" in aliases and "checkpoint_global_step_6" in aliases
