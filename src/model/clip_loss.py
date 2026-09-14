# Import PyTorch tensor operations and gradient utilities.
import torch

# Import PyTorch's base neural-network module and parameter helpers.
from torch import nn

# Import functional operations such as normalization, interpolation, and softmax.
import torch.nn.functional as F


# Make EEG and audio embeddings compatible along their temporal dimension.
def align_temporal_resolution(eeg_embedding, audio_embedding):
    # Detect sequence EEG embeddings paired with non-sequence audio vectors.
    if eeg_embedding.ndim == 3 and audio_embedding.ndim == 2:
        # Add and broadcast a time axis so each audio vector spans the EEG length.
        audio_embedding = audio_embedding.unsqueeze(-1).expand(
            # Preserve the audio batch dimension.
            -1,
            # Preserve the audio feature dimension.
            -1,
            # Match the number of temporal positions in the EEG embedding.
            eeg_embedding.shape[-1],
        )
    # Detect non-sequence EEG vectors paired with sequence audio embeddings.
    elif eeg_embedding.ndim == 2 and audio_embedding.ndim == 3:
        # Add and broadcast a time axis so each EEG vector spans the audio length.
        eeg_embedding = eeg_embedding.unsqueeze(-1).expand(
            # Preserve the EEG batch dimension.
            -1,
            # Preserve the EEG feature dimension.
            -1,
            # Match the number of temporal positions in the audio embedding.
            audio_embedding.shape[-1],
        )

    # Handle the case where both modalities already contain temporal sequences.
    if eeg_embedding.ndim == 3 and audio_embedding.ndim == 3:
        # Check whether the two sequences contain different numbers of time steps.
        if audio_embedding.shape[-1] != eeg_embedding.shape[-1]:
            # Resample audio along time so its sequence length matches the EEG.
            audio_embedding = F.interpolate(
                # Treat the audio tensor as [batch, features, time].
                audio_embedding,
                # Set the output time dimension equal to the EEG time dimension.
                size=eeg_embedding.shape[-1],
                # Use one-dimensional linear interpolation over time.
                mode="linear",
                # Avoid corner alignment to use PyTorch's standard linear sampling.
                align_corners=False,
            )
    # Return shape-compatible embeddings without changing their batch order.
    return eeg_embedding, audio_embedding


# Compute every EEG-to-audio cosine-similarity score in the batch.
def sequence_similarity_logits(eeg_embedding, audio_embedding):
    # Align vector/sequence shapes and temporal lengths before computing scores.
    eeg_embedding, audio_embedding = align_temporal_resolution(
        # Supply the EEG embeddings that will form similarity-matrix rows.
        eeg_embedding,
        # Supply the audio embeddings that will form similarity-matrix columns.
        audio_embedding,
    )

    # Use ordinary vector cosine similarity when both inputs have shape [B, D].
    if eeg_embedding.ndim == 2 and audio_embedding.ndim == 2:
        # Normalize each EEG vector to unit L2 length along its feature dimension.
        eeg_embedding = F.normalize(eeg_embedding, dim=1)
        # Normalize each audio vector to unit L2 length along its feature dimension.
        audio_embedding = F.normalize(audio_embedding, dim=1)
        # Return all pairwise dot products as a [B_eeg, B_audio] cosine matrix.
        return eeg_embedding @ audio_embedding.T

    # Use whole-sequence cosine similarity for two [B, D, T] tensors.
    if eeg_embedding.ndim == 3 and audio_embedding.ndim == 3:
        # Divide each EEG sequence by one norm computed jointly over features/time.
        eeg_embedding = eeg_embedding / eeg_embedding.norm(
            # Reduce over the feature and temporal dimensions.
            dim=(1, 2),
            # Retain singleton axes so the norm broadcasts back over each sequence.
            keepdim=True,
        # Prevent division by zero for an all-zero EEG embedding.
        ).clamp_min(1e-8)
        # Divide each audio sequence by one norm over the same two dimensions.
        audio_embedding = audio_embedding / audio_embedding.norm(
            # Reduce over the feature and temporal dimensions.
            dim=(1, 2),
            # Retain singleton axes so the norm broadcasts back over each sequence.
            keepdim=True,
        # Prevent division by zero for an all-zero audio embedding.
        ).clamp_min(1e-8)
        # Contract matching feature/time axes for every EEG/audio batch pairing.
        return torch.einsum("bct,oct->bo", eeg_embedding, audio_embedding)

    # Reject inputs that are neither two vectors nor two compatible sequences.
    raise ValueError(
        # Explain the only two embedding layouts supported by this objective.
        "EEG and audio embeddings must both be vectors [B, D] or sequences [B, D, T]."
    )


# Define the symmetric, multi-positive contrastive objective.
class CLIPLoss(nn.Module):
    """Compute symmetric CLIP/InfoNCE loss for EEG and audio embeddings."""

    # Configure the initial temperature and whether optimization can change it.
    def __init__(
        # Store methods and parameters on the current CLIPLoss instance.
        self,
        # Set the initial softmax temperature controlling logit sharpness.
        temperature=0.07,
        # Choose whether log-temperature is trainable or a fixed model buffer.
        learnable_temperature=True,
    ):
        # Initialize nn.Module so parameters and buffers are registered correctly.
        super().__init__()
        # Retain the original scalar value for configuration/introspection purposes.
        self.temperature = temperature

        # Register temperature as a trainable parameter when requested.
        if learnable_temperature:
            # Optimize log-temperature so exponentiation always yields a positive value.
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        # Otherwise retain temperature as non-trainable state in the module.
        else:
            # Register the value as a buffer so it moves devices and enters checkpoints.
            self.register_buffer(
                # Name the stored tensor consistently with the trainable alternative.
                "log_temperature",
                # Store the logarithm so both alternatives use the same forward code.
                torch.log(torch.tensor(temperature)),
            )

    # Compare a batch of EEG embeddings against a batch of audio embeddings.
    def forward(self, eeg_embedding, audio_embedding, song_ids, window_idxs):
        """
        Args:
            eeg_embedding: EEG vectors [B, D] or sequences [B, D, T].
            audio_embedding: Audio vectors [B, D] or sequences [B, D, T].
            song_ids: Integer song identifier for each batch item.
            window_idxs: Integer temporal-window identifier for each batch item.

        Returns:
            Scalar symmetric, multi-positive CLIP/InfoNCE loss.
        """

        # Convert log-temperature back to a positive temperature scalar.
        temperature = self.log_temperature.exp()
        # Compute and temperature-scale every EEG/audio pairwise similarity.
        logits = sequence_similarity_logits(eeg_embedding, audio_embedding) / temperature

        # Flatten song identifiers to one value per batch item.
        song_ids = song_ids.view(-1)
        # Flatten window identifiers to one value per batch item.
        window_idxs = window_idxs.view(-1)

        # Mark a pair as positive only when both song and window identifiers match.
        positive_mask = (
            # Compare every EEG-side song identifier with every audio-side identifier.
            (song_ids[:, None] == song_ids[None, :])
            # Require both members of the pair to refer to the same temporal window.
            & (window_idxs[:, None] == window_idxs[None, :])
        )

        # Normalize each EEG row over all candidate audio items.
        log_probs_eeg = F.log_softmax(logits, dim=1)

        # Transpose and normalize each audio row over all candidate EEG items.
        log_probs_audio = F.log_softmax(logits.T, dim=1)

        # Convert the Boolean mask to floating point for multiplication and summation.
        positive_mask_float = positive_mask.to(dtype=torch.float32)

        # Count valid audio positives for every EEG query, guarding against zero.
        positive_counts_eeg = positive_mask_float.sum(dim=1).clamp_min(1.0)
        # Count valid EEG positives for every audio query, guarding against zero.
        positive_counts_audio = positive_mask_float.T.sum(dim=1).clamp_min(1.0)

        # Compute the mean negative log probability of all audio positives per EEG.
        loss_eeg_to_audio = -(
            # Sum positive audio log probabilities and weight duplicate positives equally.
            (log_probs_eeg * positive_mask_float).sum(dim=1) / positive_counts_eeg
        # Average the per-EEG losses across the batch.
        ).mean()

        # Compute the mean negative log probability of all EEG positives per audio.
        loss_audio_to_eeg = -(
            # Sum positive EEG log probabilities and weight duplicate positives equally.
            (log_probs_audio * positive_mask_float.T).sum(dim=1)
            # Divide by the number of matching EEG positives for each audio query.
            / positive_counts_audio
        # Average the per-audio losses across the batch.
        ).mean()

        # Weight the EEG-to-audio and audio-to-EEG objectives equally.
        loss = (loss_eeg_to_audio + loss_audio_to_eeg) * 0.5

        # Return one scalar loss for gradient computation and logging.
        return loss


# Run one complete training or evaluation epoch using the selected objective.
def run_loss_epoch(
    # Yield dictionaries containing EEG, audio, and metadata tensors.
    dataloader,
    # Convert EEG tensors into audio-aligned embeddings.
    eeg_encoder,
    # Convert frozen MERT features into the shared embedding space.
    audio_projection,
    # Compute the configured EEG/audio alignment objective.
    objective,
    # Specify the CPU, CUDA, or other device used for tensor computation.
    device,
    # Supply an optimizer for training, or None for evaluation.
    optimizer=None,
):
    # Interpret the presence of an optimizer as a request to train the modules.
    is_training = optimizer is not None
    # Enable training layers such as BatchNorm updates only during optimization.
    eeg_encoder.train(mode=is_training)
    # Put the audio projection in the same training or evaluation mode.
    audio_projection.train(mode=is_training)
    # Put the selected objective in the same training or evaluation mode.
    objective.train(mode=is_training)

    # Accumulate the batch-size-weighted loss over the epoch.
    total_loss = 0.0
    # Accumulate the number of examples contributing to the epoch loss.
    total_examples = 0

    # Track gradients during training and disable them during evaluation.
    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    # Enter the selected gradient context for every batch in the epoch.
    with grad_context:
        # Iterate over batches supplied by the data loader.
        for batch in dataloader:
            # Convert EEG samples to float tensors and move them to the target device.
            eeg = batch["eeg"].float().to(device)
            # Convert audio features to float tensors and move them to the target device.
            audio_features = batch["audio"].float().to(device)

            # Convert subject identifiers to device-resident integer index tensors.
            subject_ids = batch["subject_id"].long().to(device)
            # Convert song identifiers to device-resident integer index tensors.
            song_ids = batch["song_id"].long().to(device)
            # Convert window identifiers to device-resident integer index tensors.
            window_idxs = batch["window_idx"].long().to(device)

            # Encode EEG while optionally conditioning on each sample's subject ID.
            eeg_embed = eeg_encoder(eeg, subject_ids)
            # Project precomputed audio features into the model's alignment space.
            audio_embed = audio_projection(audio_features)
            # Ensure both modalities have compatible temporal dimensions.
            eeg_embed, audio_embed = align_temporal_resolution(eeg_embed, audio_embed)

            # Calculate the selected scalar alignment loss for this batch.
            loss = objective(eeg_embed, audio_embed, song_ids, window_idxs)

            # Update learnable parameters only when an optimizer was provided.
            if is_training:
                # Clear gradients left over from the previous batch.
                optimizer.zero_grad()
                # Backpropagate the loss through all trainable model components.
                loss.backward()
                # Apply one optimizer update using the newly computed gradients.
                optimizer.step()

            # Read the number of EEG examples in the current batch.
            batch_size = eeg.size(0)

            # Add this batch's loss weighted by its number of examples.
            total_loss += loss.item() * batch_size
            # Add this batch's example count to the running denominator.
            total_examples += batch_size

        # Return the example-weighted mean loss across all processed batches.
        return {"loss": total_loss / total_examples}
