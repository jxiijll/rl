from __future__ import annotations

import torch


def _attention_heads(hidden_dim: int) -> int:
    for heads in (4, 2):
        if hidden_dim % heads == 0:
            return heads
    return 1


class StateEncoder(torch.nn.Module):
    def __init__(self, state_feature_dim: int, hidden_dim: int):
        super().__init__()
        self.state_feature_dim = int(state_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.state_feature_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        if state_features.dim() != 1:
            raise ValueError("state_features must have shape [state_feature_dim]")
        return self.net(state_features)


class QueueSetEncoder(torch.nn.Module):
    def __init__(self, queue_feature_dim: int, hidden_dim: int):
        super().__init__()
        self.queue_feature_dim = int(queue_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.input = torch.nn.Sequential(
            torch.nn.Linear(self.queue_feature_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=_attention_heads(self.hidden_dim),
            batch_first=True,
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 2),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        queue_features: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if queue_features.dim() == 2:
            queue_features = queue_features.unsqueeze(0)
            squeeze_batch = True
        elif queue_features.dim() == 3:
            squeeze_batch = False
        else:
            raise ValueError("queue_features must have shape [queue_size, queue_feature_dim] or [batch, queue_size, queue_feature_dim]")

        batch_size, queue_size, _feature_dim = queue_features.shape
        if queue_size == 0:
            context = torch.zeros(batch_size, self.hidden_dim, dtype=queue_features.dtype, device=queue_features.device)
            return context.squeeze(0) if squeeze_batch else context

        encoded = self.input(queue_features)
        attended, _ = self.attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if key_padding_mask is None:
            mean_pool = attended.mean(dim=1)
            max_pool = attended.max(dim=1).values
        else:
            valid = (~key_padding_mask).unsqueeze(-1).to(attended.dtype)
            denom = valid.sum(dim=1).clamp_min(1.0)
            masked = attended * valid
            mean_pool = masked.sum(dim=1) / denom
            max_pool = attended.masked_fill(key_padding_mask.unsqueeze(-1), -torch.inf).max(dim=1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        context = self.output(torch.cat([mean_pool, max_pool], dim=-1))
        return context.squeeze(0) if squeeze_batch else context


class GRUFairPolicy(torch.nn.Module):
    def __init__(
        self,
        job_feature_dim: int,
        state_feature_dim: int,
        hidden_dim: int = 32,
        queue_feature_dim: int | None = None,
    ):
        super().__init__()
        self.job_feature_dim = int(job_feature_dim)
        self.state_feature_dim = int(state_feature_dim)
        self.queue_feature_dim = int(queue_feature_dim or job_feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.state_encoder = StateEncoder(self.state_feature_dim, self.hidden_dim)
        self.queue_encoder = QueueSetEncoder(self.queue_feature_dim, self.hidden_dim)
        self.gru_input = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 2),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.job_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.job_feature_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.score_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 6),
            torch.nn.Linear(self.hidden_dim * 6, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.609))

    def initial_hidden(self, device: torch.device | None = None) -> torch.Tensor:
        first_param = next(self.parameters())
        return torch.zeros(1, 1, self.hidden_dim, device=device or first_param.device)

    def forward(
        self,
        job_features: torch.Tensor,
        state_features: torch.Tensor,
        hidden: torch.Tensor,
        queue_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if job_features.dim() != 2:
            raise ValueError("job_features must have shape [num_candidates, job_feature_dim]")

        encoded_state = self.state_encoder(state_features)
        queue_context = self.queue_encoder(queue_features if queue_features is not None else job_features)
        gru_features = self.gru_input(torch.cat([encoded_state, queue_context], dim=-1))
        _, new_hidden = self.gru(gru_features.view(1, 1, self.hidden_dim), hidden)
        context = new_hidden[-1, 0].expand(job_features.shape[0], -1)
        encoded_jobs = self.job_encoder(job_features)
        set_mean = encoded_jobs.mean(dim=0, keepdim=True).expand_as(encoded_jobs)
        set_max = encoded_jobs.max(dim=0, keepdim=True).values.expand_as(encoded_jobs)
        queue_context_expanded = queue_context.expand_as(encoded_jobs)
        interactions = encoded_jobs * context
        scores = self.score_mlp(
            torch.cat([encoded_jobs, context, interactions, set_mean, set_max, queue_context_expanded], dim=1)
        ).squeeze(-1)
        scores = scores * torch.clamp(self.logit_scale.exp(), min=1.0, max=20.0)
        return scores, new_hidden


class GRUValuePolicy(torch.nn.Module):
    def __init__(
        self,
        state_feature_dim: int,
        hidden_dim: int = 64,
        queue_feature_dim: int | None = None,
    ):
        super().__init__()
        self.state_feature_dim = int(state_feature_dim)
        self.queue_feature_dim = int(queue_feature_dim or state_feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.state_encoder = StateEncoder(self.state_feature_dim, self.hidden_dim)
        self.queue_encoder = QueueSetEncoder(self.queue_feature_dim, self.hidden_dim)
        self.gru_input = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 2),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.value_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def initial_hidden(self, device: torch.device | None = None) -> torch.Tensor:
        first_param = next(self.parameters())
        return torch.zeros(1, 1, self.hidden_dim, device=device or first_param.device)

    def forward(
        self,
        state_features: torch.Tensor,
        hidden: torch.Tensor,
        queue_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_state = self.state_encoder(state_features)
        if queue_features is None:
            queue_features = torch.zeros(
                0,
                self.queue_feature_dim,
                dtype=state_features.dtype,
                device=state_features.device,
            )
        queue_context = self.queue_encoder(queue_features)
        gru_features = self.gru_input(torch.cat([encoded_state, queue_context], dim=-1))
        _, new_hidden = self.gru(gru_features.view(1, 1, self.hidden_dim), hidden)
        value = self.value_head(new_hidden[-1, 0]).squeeze(-1)
        return value, new_hidden
