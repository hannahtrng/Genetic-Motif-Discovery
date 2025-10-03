import numpy as np
import random
from multiprocessing import Pool

# parse FASTA files
def parse_fasta(filename):
    print(f"Parsing FASTA file: {filename}")
    sequences = {}
    with open(filename, 'r') as file:
        seq_id = None
        seq = ""
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = seq
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq
    return sequences

# find the reverse complement of a sequence
def reverse_complement(sequence):
    complement = str.maketrans('ACGT', 'TGCA')
    return sequence.translate(complement)[::-1]

# computes background frequencies for a set of sequences
def compute_background_frequencies(sequences):
    counts = np.zeros(4)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    total = 0
    for seq in sequences.values():
        for base in seq:
            if base in mapping:
                counts[mapping[base]] += 1
                total += 1
    return counts / total if total > 0 else counts

# builds a position weight matrix (PWM) from a set of motifs
def build_pwm(motifs, k, bg_freq=None):
    pwm = np.ones((4, k))
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    if bg_freq is None:
        bg_freq = np.ones(4) / 4
    
    for motif in motifs:
        for i, base in enumerate(motif):
            pwm[mapping[base], i] += 1
    
    pwm /= pwm.sum(axis=0)
    pwm = np.log2((pwm + 1e-6) / (bg_freq[:, None] + 1e-6))
    return pwm

# calculates motif score using a PWM
def motif_score(motif, pwm):
    score = 0
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, base in enumerate(motif):
        score += pwm[mapping[base], i]
    return score

# performs expectation-maximization (EM) motif search
def em_motif_search(sequences, k, max_iterations=100):
    print(f"Starting EM motif search with k={k} for {max_iterations} iterations")
    motifs = [random.choice([seq[i:i+k] for i in range(len(seq)-k+1)]) for seq in sequences.values()]
    
    for _ in range(max_iterations):
        pwm = build_pwm(motifs, k)
        motifs = [max([seq[i:i+k] for i in range(len(seq)-k+1)], key=lambda m: motif_score(m, pwm)) for seq in sequences.values()]
    
    return build_pwm(motifs, k)

# predicts the summit of a sequence using a PWM
def predict_summit(sequence, pwm, k):
    best_score = -float('inf')
    best_pos = len(sequence) // 2
    for i in range(len(sequence) - k + 1):
        motif = sequence[i:i+k]
        rev_motif = reverse_complement(motif)
        score = motif_score(motif, pwm)
        rev_score = motif_score(rev_motif, pwm)
        
        if score > best_score:
            best_score = score
            best_pos = i + 1
        if rev_score > best_score:
            best_score = rev_score
            best_pos = i + 1
    
    return max(31, min(171, best_pos))

# parallelizes the summit prediction process
def parallel_summit_prediction(sequences, pwm, k):
    with Pool() as pool:
        results = pool.starmap(predict_summit, [(seq, pwm, k) for seq in sequences.values()])
    return dict(zip(sequences.keys(), results))

def eval_predictions(predictions, true_summits):
    print("Validating predictions against true summits")
    true_val = sum(1 for seq_id, pos in predictions.items() if abs(pos - true_summits.get(seq_id, -1)) <= 15)
    accuracy = true_val / len(predictions) if predictions else 0
    print(f"Accuracy: {accuracy:.2%} ({true_val}/{len(predictions)})")
    return accuracy

def main():

    # parse files: CHANGE IF NEEDED
    centered_seqs = parse_fasta("boundcentered.fasta")
    random_seqs = parse_fasta("boundrandomoffset.fasta")
    
    k = 10  # motif length
    pwm = em_motif_search(centered_seqs, k)
    predictions = parallel_summit_prediction(random_seqs, pwm, k)
    centered_predictions = parallel_summit_prediction(centered_seqs, pwm, k)

    # test for boundcentered.fasta (accuracy)
    true_summits = {seq_id: len(seq) // 2 for seq_id, seq in centered_seqs.items()}  
    eval_predictions(centered_predictions, true_summits)
    
    with open("predictions.csv", "w") as f:
        for seq_id, pos in predictions.items():
            f.write(f"{seq_id},{pos}\n")
    
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()
