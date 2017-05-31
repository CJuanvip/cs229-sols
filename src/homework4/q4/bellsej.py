import numpy            as np
import scipy.io.wavfile as wav


# The sampling freequency of the audio data.
Fs = 11025


def load_data(handle):
    return np.loadtxt(handle)

def normalize_mix(mix):
    rows = mix.shape[0]

    return 0.99 * mix / (np.ones((rows,1)) * np.max(np.abs(mix)))


def split_and_normalize_mixes(mixes):
    nsamples, mix_count = mixes.shape

    for i in range(mix_count):
        normalized_mix = normalize_mix(mixes[:, i])
        mix_name = 'mix{}.wav'.format(i)
        wav.write(mix_name, Fs, normalized_mix)

    return nsamples, mix_count


def analyze(mixes):
    """
    Run ICA on the mixes to separate them out.
    """
    nsamples, mix_count = mixes.shape
    anneal = [0.1,  0.1,  0.1,   0.05,  0.05,  0.05,  0.02,  0.02, 
              0.01, 0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001];

    W = np.eye(mix_count)

    for round in range(len(anneal)):
        m = mix.shape[0]
        order = np.permutation(m)
        for i in range(m):
            x = mix[order[i], :].T
            g = 1 / (1 + np.exp(-W.dot(x)))
            W = W + anneal[round] * ((1 - 2 * g).dot(x.T.dot(np.inv(W.T))))

    return W


def normalize_source(source):
    rows = mix.shape[0]

    return 0.99 * source / (np.ones((rows,1)) * np.max(np.abs(source)))


def write_sources(sources):
    nsamples, source_count = sources.shape

    for i in range(source_count):
        normalized_source = normalize_source(sources[:, i])
        source_name = 'unmix{}.wav'.format(i)
        wav.write(source_name, Fs, normalized_source)

    return nsamples, source_count