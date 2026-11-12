import torch

from probe.losses import per_answer_bce_loss
from probe.mock_data import generate_mock_dataset
from probe.model import MLPProbe


def test_mock_dataset_shapes():
    data = generate_mock_dataset(n_prompts=20, feature_dim=16, num_layers=3, min_tokens=3, max_tokens=5, seed=0)
    n_rows = data.features.shape[0]
    assert data.features.shape == (n_rows, 16, 3)
    assert data.labels.shape == (n_rows,)
    assert data.prompt_ids.shape == (n_rows,)
    assert data.token_positions.shape == (n_rows,)
    assert set(data.labels.unique().tolist()) <= {0.0, 1.0}


def test_every_token_in_a_prompt_shares_its_label():
    data = generate_mock_dataset(n_prompts=30, feature_dim=8, num_layers=3, seed=0)
    for prompt_id in data.prompt_ids.unique().tolist():
        rows = data.prompt_ids == prompt_id
        assert data.labels[rows].unique().numel() == 1


def test_token_positions_are_contiguous_per_prompt():
    data = generate_mock_dataset(n_prompts=10, feature_dim=8, num_layers=3, min_tokens=4, max_tokens=4, seed=0)
    for prompt_id in data.prompt_ids.unique().tolist():
        positions = data.token_positions[data.prompt_ids == prompt_id]
        assert positions.tolist() == list(range(4))


def test_scalar_mix_output_dim_independent_of_num_layers():
    model_3 = MLPProbe(feature_dim=16, num_layers=3, fusion="scalar_mix")
    model_5 = MLPProbe(feature_dim=16, num_layers=5, fusion="scalar_mix")
    assert model_3.fusion.output_dim == model_5.fusion.output_dim == 16


def test_concat_output_dim_scales_with_num_layers():
    model = MLPProbe(feature_dim=16, num_layers=3, fusion="concat")
    assert model.fusion.output_dim == 48


def test_forward_shapes_both_fusions():
    x = torch.randn(4, 16, 3)
    for fusion in ("concat", "scalar_mix"):
        model = MLPProbe(feature_dim=16, num_layers=3, fusion=fusion)
        logits = model(x)
        assert logits.shape == (4,)
        proba = model.predict_proba(x)
        assert torch.all((proba >= 0) & (proba <= 1))


def test_scalar_mix_layer_weights_sum_to_one():
    model = MLPProbe(feature_dim=16, num_layers=3, fusion="scalar_mix")
    weights = model.fusion.layer_weights()
    assert weights.shape == (3,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_per_answer_loss_weights_prompts_equally_regardless_of_length():
    # Two prompts, one with 1 row one with 3 rows, both wrong by the same margin.
    logits = torch.tensor([5.0, 5.0, 5.0, 5.0])
    labels = torch.tensor([0.0, 0.0, 0.0, 0.0])
    group_ids = torch.tensor([0, 1, 1, 1])
    per_answer = per_answer_bce_loss(logits, labels, group_ids)
    flat_mean = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    # every row has identical loss here, so per-answer and flat mean coincide ...
    assert torch.isclose(per_answer, flat_mean, atol=1e-5)

    # ... but if the long prompt's rows are easy (correct, confident) and the
    # short prompt's row is hard (wrong, confident), a flat mean is dominated
    # by the three easy rows; per-answer isn't.
    logits = torch.tensor([5.0, -5.0, -5.0, -5.0])  # group 0's only row is wrong
    per_answer = per_answer_bce_loss(logits, labels, group_ids)
    flat_mean = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert per_answer > flat_mean


def test_training_learns_signal_above_chance_with_prompt_level_split():
    data = generate_mock_dataset(n_prompts=300, feature_dim=32, num_layers=3, seed=1)
    n_train_prompts = 240
    train_mask = data.prompt_ids < n_train_prompts
    val_mask = ~train_mask
    train_x, train_y = data.features[train_mask], data.labels[train_mask]
    val_x, val_y = data.features[val_mask], data.labels[val_mask]
    train_prompt_ids = data.prompt_ids[train_mask]

    model = MLPProbe(feature_dim=32, num_layers=3, fusion="scalar_mix")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(80):
        optimizer.zero_grad()
        loss = per_answer_bce_loss(model(train_x), train_y, train_prompt_ids)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_acc = ((model.predict_proba(val_x) > 0.5).float() == val_y).float().mean().item()
    assert val_acc > 0.6
