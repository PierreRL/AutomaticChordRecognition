"""
A custom Hidden Markov smoothing chord sequences.
"""

import autorootcwd

import torch
import torch.nn.functional as F

class HMMSmoother(torch.nn.Module):
    """
    An HMM-based smoother for chord probability sequences using the forward–backward algorithm.
    This will yield a per-frame posterior distribution over chords that incorporates
    transition probabilities and all frames (past + future).
    """
    def __init__(self, num_classes: int, alpha: float = 0.2):
        """
        Args:
            num_classes (int): Number of chord classes in the vocabulary.
            alpha (float): Probability of staying in the same chord (diagonal). 
                           Off-diagonal transitions are uniform.
                           0.5 <= alpha < 1.0 is a typical range.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Create the transition matrix (num_classes x num_classes)
        # Diagonal = alpha, off-diagonal = (1 - alpha)/(num_classes - 1)
        self.transition_matrix = self._create_transition_matrix(alpha)
        
        # Simple choice of uniform initial distribution over chords
        self.initial_distribution = torch.full((num_classes,), 1.0 / num_classes)

    def _create_transition_matrix(self, alpha: float) -> torch.Tensor:
        """
        Creates a transition matrix with self-transition probability alpha
        and uniform off-diagonal probability.
        """
        mat = torch.full((self.num_classes, self.num_classes),
                         (1.0 - alpha) / (self.num_classes - 1))
        # Fill the diagonal with alpha
        for i in range(self.num_classes):
            mat[i, i] = alpha
        return mat
    
    def forward(self, logits: torch.Tensor, device = None) -> torch.Tensor:
        """
        Applies the forward–backward algorithm to produce, at each frame,
        a probability distribution over all chord classes.
        A bit like Viterbi, but instead of finding the most likely sequence,
        we compute the probability of each chord at each frame given all frames both forward and backward.

        Args:
            logits (torch.Tensor): Tensor of shape (B, T, num_classes), where B is
                                   the number of sequences in the batch, T is the number
                                   of frames, and C is the number of classes. These are
                                   raw, unnormalized logits for each frame.

        Returns:
            posterior (torch.Tensor): Shape (B, T, num_classes)
                                      posterior[b, t, i] = P(chord=i at time t | all frames).
        """
        eps = 1e-12
        if device is None:
            device = logits.device

        # Convert transition matrix and initial distribution to log space
        log_transition = torch.log(self.transition_matrix + eps).to(device)    # (C, C)
        log_init = torch.log(self.initial_distribution + eps).to(device)       # (C,)
        logits = logits.to(device)  # (B, T, C)

        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, C)

        B, T, C = log_probs.shape

        # Prepare alpha and beta
        alpha = torch.zeros((B, T, C), device=device)
        beta = torch.zeros((B, T, C), device=device)

        # Initialize alpha at time 0
        #   alpha[b, 0, :] = log_init + log_probs[b, 0, :]
        alpha[:, 0, :] = log_init.unsqueeze(0) + log_probs[:, 0, :]  # (B, C)

        # Forward pass
        # alpha[t] = logsumexp( alpha[t-1] + transition_matrix ) + log_emiss[t]
        for t in range(1, T):
            # shape of alpha[:, t-1, :]: (B, C)
            # shape of log_transition: (C, C)
            # we want to add alpha[:, t-1, i] to transition[i, j] and sum over i
            # => broadcast to (B, C, C) then logsumexp(dim=1) to get shape (B, C)
            prev_alpha = alpha[:, t-1, :].unsqueeze(2)  # (B, C, 1)
            tmp = prev_alpha + log_transition.unsqueeze(0)  # (B, C, C)
            alpha[:, t, :] = torch.logsumexp(tmp, dim=1) + log_probs[:, t, :]

        # Backward pass
        # beta[t] = logsumexp( transition_matrix + log_emiss[t+1] + beta[t+1], dim=1 )
        # Note we compute from T-2 down to 0
        for t in range(T - 2, -1, -1):
            # shape of beta[:, t+1, :]: (B, C)
            # log_emiss[t+1]: (B, C)
            next_beta = beta[:, t+1, :].unsqueeze(1)  # (B, 1, C)
            next_emiss = log_probs[:, t+1, :].unsqueeze(1)  # (B, 1, C)
            # Combine with transition_matrix (C, C) => shape (B, C, C) after broadcasting
            tmp = log_transition.unsqueeze(0) + next_emiss + next_beta  # (B, C, C)
            beta[:, t, :] = torch.logsumexp(tmp, dim=2)  # sum over next-state dimension

        # Combine alpha and beta to get posterior in log space
        posterior_log = alpha + beta  # shape (B, T, C)

        # Normalize in log space per-frame
        # We want softmax over the chord-dimension (C) for each (B, t)
        # => logsumexp(..., dim=-1)
        posterior_log = posterior_log - torch.logsumexp(posterior_log, dim=2, keepdim=True)

        # Exponentiate to get posterior
        posterior = torch.exp(posterior_log)
        return posterior
    

# Example usage
def main():
    # Create a dummy sequence of chord probabilities
    B, T, C = 8, 4000, 170
    chord_logits = torch.randn(B, T, C)

    # Create the smoother
    smoother = HMMSmoother(num_classes=C, alpha=0.9)

    # Apply the smoother
    posterior = smoother(chord_logits)
    print(posterior.shape)  # (B, T, C)
    print(posterior[0, 0])  # Posterior distribution over chords at time 0 for the first sequence

    # Check that the posterior sums to 1
    print(posterior[0].sum(dim=-1))  # Should be close to 1

if __name__ == "__main__":
    main()