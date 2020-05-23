## MIT License

import numpy as np
import scipy, scipy.io, scipy.io.wavfile, scipy.signal
import scipy.interpolate
import os
import sys
from pathlib import Path
import argparse
from numpy.polynomial import polynomial as P

# This function is copied directly from https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py
# Copyright (c) 2008 Cournapeau David
# (MIT licensed)
def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1 / r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order + 1, r.dtype)
    # temporary array
    t = np.empty(order + 1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k[i - 1] = -acc / e
        a[i] = k[i - 1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i - 1] * np.conj(t[i - j])

        e *= 1 - k[i - 1] * np.conj(k[i - 1])

    return a, e, k





def lsp_to_lpc(lsp):
    """Convert line spectral pairs to LPC"""
    ps = np.concatenate((lsp[:, 0], -lsp[::-1, 0], [np.pi]))
    qs = np.concatenate((lsp[:, 1], [0], -lsp[::-1, 1]))

    p = np.cos(ps) - np.sin(ps) * 1.0j
    q = np.cos(qs) - np.sin(qs) * 1.0j

    p = np.real(P.polyfromroots(p))
    q = -np.real(P.polyfromroots(q))

    a = 0.5 * (p + q)
    return a[:-1]



def lpc_to_lsp(lpc):
    """Convert LPC to line spectral pairs"""
    l = len(lpc) + 1
    a = np.zeros((l,))
    a[0:-1] = lpc
    p = np.zeros((l,))
    q = np.zeros((l,))
    for i in range(l):
        j = l - i - 1
        p[i] = a[i] + a[j]
        q[i] = a[i] - a[j]

    ps = np.sort(np.angle(np.roots(p)))
    qs = np.sort(np.angle(np.roots(q)))
    lsp = np.vstack([ps[: len(ps) // 2], qs[: len(qs) // 2]]).T
    return lsp


def lpc_to_formants(lpc, sr):
    """Convert LPC to formants    
    """

    # extract roots, get angle and radius
    roots = np.roots(lpc)

    pos_roots = roots[np.imag(roots) >= 0]
    if len(pos_roots) < len(roots) // 2:
        pos_roots = list(pos_roots) + [0] * (len(roots) // 2 - len(pos_roots))
    if len(pos_roots) > len(roots) // 2:
        pos_roots = pos_roots[: len(roots) // 2]

    w = np.angle(pos_roots)
    a = np.abs(pos_roots)

    order = np.argsort(w)
    w = w[order]
    a = a[order]

    freqs = w * (sr / (2 * np.pi))
    bws = -0.5 * (sr / (2 * np.pi)) * np.log(a)

    # exclude DC and sr/2 frequencies
    return freqs, bws





def lpc(wave, order):
    """Compute LPC of the waveform. 
    a: the LPC coefficients
    e: the total error
    k: the reflection coefficients
    
    Typically only a is required.
    """
    # only use right half of autocorrelation, normalised by total length
    autocorr = scipy.signal.correlate(wave, wave)[len(wave) - 1 :] / len(wave)
    a, e, k = levinson_1d(autocorr, order)
    return a, e, k


def lpc_vocode(
    wave,
    frame_len,
    order,
    carrier,
    residual_amp=0.0,
    vocode_amp=1.0,
    env=False,
    freq_shift=1.0,
):
    """
    Apply LPC vocoding to a pair of signals using 50% overlap-add Hamming window resynthesis
    The modulator `wave` is applied to the carrier `imposed`
    
    Parameters:
    ---
    wave: modulator wave
    frame_len: length of frames
    order: LPC order (typically 2-30)
    carrier: carrier signal; should be at least as long as wave
    residual_amp: amplitude of LPC residual to include in output
    vocode_amp: amplitude of vocoded signal 
    env: if True, the original volume envelope of wave is imposed on the output
          otherwise, no volume modulation is applied
    freq_shift: (default 1.0) shift the frequency of the resonances by the given scale factor. Warning :
        values >1.1 are usually unstable, and values <0.5 likewise.
    """

    # precompute the hamming window
    window = scipy.signal.hann(frame_len)
    t = np.arange(frame_len)
    # allocate the array for the output
    vocode = np.zeros(len(wave + frame_len))
    last = np.zeros(order)
    # 50% window steps for overlap-add
    for i in range(0, len(wave), frame_len // 2):
        # slice the wave
        wave_slice = wave[i : i + frame_len]
        carrier_slice = carrier[i : i + frame_len]
        if len(wave_slice) == frame_len:
            # compute LPC
            a, error, reflection = lpc(wave_slice, order)

            # apply shifting in LSP space
            lsp = lpc_to_lsp(a)
            lsp = (lsp * freq_shift + np.pi) % (np.pi) - np.pi
            a = lsp_to_lpc(lsp)

            # compute the LPC residual
            residual = scipy.signal.lfilter(a, 1.0, wave_slice)
            # filter, using LPC as the *IIR* component
            # vocoded, last = scipy.signal.lfilter([1.], a, carrier_slice, zi=last)
            vocoded = scipy.signal.lfilter([1.0], a, carrier_slice)

            # match RMS of original signal
            if env:
                voc_amp = 1e-5 + np.sqrt(np.mean(vocoded ** 2))
                wave_amp = 1e-5 + np.sqrt(np.mean(wave_slice ** 2))
                vocoded = vocoded * (wave_amp / voc_amp)

            # Hann window 50%-overlap-add to remove clicking
            vocode[i : i + frame_len] += (
                vocoded * vocode_amp + residual * residual_amp
            ) * window

    return vocode[: len(wave)]



def get_lsp(wave, frame_len, order, sr=44100, overlap=None):
    """Get the LSP and envelope of the given wave form.    
    Parameters:
    wave:  Signal to analyse, as a 1D matrix
    frame_len: Length of analysis window, in samples
    order: Order of the LPC analysis performed
    sr: Sample rate, in Hz        
    """
    overlap = overlap or frame_len // 2
    times, env, lsps = [], [], []
    for i in range(0, len(wave), overlap):        
        wave_slice = wave[i : i + frame_len]
        if len(wave_slice) == frame_len:
            # compute LPC
            a, error, reflection = lpc(wave_slice, order)
            # use LSP (freq from mean angle, bw from spacing)                        
            lsps.append(lpc_to_lsp(a))
            times.append(i / float(sr))
            env.append(np.sqrt(np.mean(wave_slice ** 2)))

    return np.array(times), np.array(lsps), np.array(env)

def formants_from_lsp(lsps, sr):
    fr = sr / (2*np.pi)
    freqs = -np.mean(lsps, axis=-1) * fr
    bws = 0.5 * np.diff(lsps, axis=-1)[..., 0] * fr    
    return freqs, bws

import scipy.ndimage
def sinethesise(wave, frame_len, order, sr=44100, use_lsp=False, noise=1.0, overlap=None):
    overlap = overlap or 0.5
    frame_overlap = int(frame_len * overlap)
    times, lsps, env_rms = get_lsp(wave, frame_len, order, sr, frame_overlap)
   
    formants, formant_bw = formants_from_lsp(lsps, sr)

    synthesize = np.zeros_like(wave)
    window = scipy.signal.hann(frame_len)
    t = np.arange(0.0, frame_len)
    k = 0    
    for i in range(0, len(wave), frame_overlap):

        if len(synthesize[i : i + frame_len]) == frame_len:
            # noise component
            syn_slice = np.zeros_like(t)

            # resonances
            for band in range(formants.shape[1]):
                freq = formants[k, band]
                bw = formant_bw[k, band]
                amp = np.exp(bw/60.0)  # weight sines by inverse bandwidth                
                if freq>90.0:
                    syn_slice += np.sin(freq * (t + i) / (sr / (2 * np.pi))) * amp

            synthesize[i : i + frame_len] += window * syn_slice * env_rms[k]
        k += 1
    return synthesize

def sinethesise_alternative(wave, frame_len, order, sr=44100, use_lsp=False, noise=1.0, overlap=None):
    overlap = overlap or 0.5
    frame_overlap = int(frame_len * overlap)
    times, lsps, env_rms = get_lsp(wave, frame_len, order, sr, frame_overlap)
    t = np.arange(len(wave)) / sr
    synthesize = np.zeros_like(wave)
    n_bands = lsps.shape[1]
    env_int = scipy.interpolate.interp1d(times, env_rms, fill_value='extrapolate', kind='cubic')    
    lsp_int = [scipy.interpolate.interp1d(times, lsps[:,band,:], axis=0, fill_value='extrapolate', kind='nearest') for band in range(n_bands)]
    env = env_int(t)

    for band in range(n_bands):
        lsp_smooth = lsp_int[band](t)
        freq, bw = formants_from_lsp(lsp_smooth, sr)    
        amp = np.exp(bw/60.0)
        synthesize += np.sin(freq * t * (2 * np.pi)) * amp
    
    return synthesize * env


def load_wave(fname):
    """Load a 16 bit wave file and return normalised in 0,1 range.
    Convert stereo WAV to mono by simple averaging. """
    # load and return a wave file
    sr, wave = scipy.io.wavfile.read(fname)
    # convert to mono
    if len(wave.shape) > 1:
        wave = np.mean(wave, axis=1)
    return wave / 32768.0, sr

def modfm_buzz(samples, f, sr, k):
    """Generate a pulse train using modfm:
        y(t) = cos(x(t)) * exp(cos(x(t))*k - k)
        
        samples: number of samples to generate
        f: base frequency (Hz)
        sr: sample rate (Hz)
        k: modulation depth; higher has more harmonics but increases risk of aliasing
        (e.g. k=1000 for f=50, k=100 for f=200, k=2 for f=4000)        
    
    """
    t = np.arange(samples)
    phase = f * 2 * np.pi * (t / float(sr))
    # simple pulse oscillator (ModFM)
    buzz = np.cos(phase) * np.exp(np.cos(phase) * k - k)
    return buzz



def bp_filter_and_decimate(x, low, high, fs, decimate=1):
    b, a = scipy.signal.butter(4, Wn=[low, high], btype="band", fs=fs)
    decimated = scipy.signal.filtfilt(b, a, x)[::decimate]
    # pre-emphasis    
    decimated = scipy.signal.filtfilt([1.0], np.array([1.0, 0.63]), decimated)
    return decimated

def normalize(x):
    return 0.9 * (x / np.max(x) )


def upsample(x, factor):
    return scipy.signal.resample_poly(x, factor, 1)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_wav", help="The input file, as a WAV file; ideally 44.1KHz mono."
    )
    parser.add_argument(
        "output_wav",
        nargs="?",
        help="The output file to write to; defaults to <input>_sws.wav",
        default=None,
    )
    parser.add_argument("--low", help="Lowpass filter cutoff", type=float, default=150)
    parser.add_argument("--high", help="Highpass filter cutoff", type=float, default=3000)
    parser.add_argument(
        "--order", "-o", help="Number of components in synthesis", default=4, type=int
    )
   
    parser.add_argument(
        "--decimate", "-d", help="Sample rate decimation before analysis", default=8, type=int
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        help="LPC window size; smaller means faster changing signal; larger is smoother",
        default=250,
    )
    parser.add_argument(
        "--interpolate",
        "-i",
        help="Enable interpolation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sine",
        "-s",
        help="Resynthesise using sinewave speech (default)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--buzz",
        "-b",
        help="Resynthesie using buzz at given frequency (Hz)",
        default=None,
    )
    parser.add_argument(
        "--noise", "-n", help="Resynthesize using filtered white noise", action="store_true"
    )

    parser.add_argument(
        "--overlap", "-l", help="Window overlap, as fraction of the window length", default=0.5, type=float,
    )

    args = parser.parse_args(args[1:])

    args.output_wav = (
        args.output_wav or os.path.splitext(args.input_wav)[0] + "_sws.wav"
    )

    input_path = Path(args.input_wav)
    output_path = Path(args.output_wav)

    if not input_path.exists():
        print(f"Cannot open {args.input_wav} for reading.")
        exit(-1)

    
    wav, fs = load_wave(input_path)
    print(f"Read {input_path}")

    wav_filtered = normalize(bp_filter_and_decimate(
        wav, args.low, args.high, fs, decimate=args.decimate
    ))
    order = 2 * args.order + 2
    if args.sine:
        
            modulated = sinethesise(
                wav_filtered,
                frame_len=args.window,
                order=order,
                use_lsp=True,
                sr=fs / args.decimate,
                noise=0.0,
                overlap=args.overlap
            )
    if args.buzz or args.noise:

        if args.buzz:
            N = 12 * np.log2(float(args.buzz)/440.0) + 69
            
            k = np.exp(-0.1513*N) + 15.927 # ModFM k values from: http://mural.maynoothuniversity.ie/4104/1/VL_New_perspectives.pdf
            
            carrier = modfm_buzz(len(wav_filtered), f=np.full(len(wav_filtered), args.buzz, dtype=np.float64),
                        sr=float(fs/args.decimate), k=np.full(len(wav_filtered), k*k))
        if args.noise:
            carrier = np.random.normal(0,1,len(wav_filtered))

        modulated = lpc_vocode(wav_filtered, frame_len=args.window, order=order,
            carrier=carrier, residual_amp=0, vocode_amp=1, env=True, freq_shift=1)

    # un-decimate, normalize and write out
    up_modulated = normalize(upsample(modulated, args.decimate))
    
    scipy.io.wavfile.write(output_path, fs, (up_modulated*32767.0).astype(np.int16))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main(sys.argv)

