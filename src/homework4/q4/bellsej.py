import numpy            as np
import scipy.io.wavfile as wav


# The sampling frequency of the audio data.
Fs = 11025


def load_data(handle):
    return np.loadtxt(handle)

def normalize_mix(mix):
    return 0.99 * mix / (np.ones(mix.shape[0]) * np.max(np.abs(mix)))

def normalize_mixes(mixes):
    mix_count = mixes.shape[1]
    normalized_mixes = np.zeros(mixes.shape)

    for i in range(mix_count):
        mixes[:, i] = normalize_mix(mixes[:, i])

    return normalized_mixes


def split_and_normalize_mixes(mixes):
    mix_count = mixes.shape[1]
    normalized_mixes = np.zeros(mixes.shape)

    for i in range(mix_count):
        normalized_mixes[:, i] = normalize_mix(mixes[:, i])
        mix_name = 'mix{}.wav'.format(i)
        wav.write(mix_name, Fs, normalized_mixes[:, i])

    return normalized_mixes


def analyze(mixes):
    """
    Run ICA on the mixes to separate them out.
    """
    nsamples, mix_count = mixes.shape
    anneal = [0.1,  0.1,  0.1,   0.05,  0.05,  0.05,  0.02,  0.02, 
              0.01, 0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001];

    W = np.eye(mix_count)

    for round in range(len(anneal)):
        m = mixes.shape[0]
        order = np.random.permutation(m)
        for i in range(m):
            x = mixes[order[i], :].T
            g = 1 / (1 + np.exp(-W.dot(x)))
            W = W + anneal[round] * (np.outer((1 - 2 * g), x.T) + np.linalg.inv(W.T))

    return W

def write_mixes(unmixes):
    nsamples, unmix_count = unmixes.shape
    
    for i in range(unmix_count):
        unmix_name = 'unmix{}.wav'.format(i)
        wav.write(unmix_name, Fs, unmixes[:, i])
