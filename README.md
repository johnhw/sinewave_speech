# Sinewave Speech
<img src="imgs/sws.png" width="100%">

A Python implementation of sinewave speech. Converts WAV files of human-speech to [sinewave speech](https://en.wikipedia.org/wiki/Sinewave_synthesis) using linear predictive coding (LPC). This is a "simplified" representation of the speech with a small number of frequency and amplitude modulated sine waves. It is surprising how much remains intelligible after this transformation.

Listen to some sounds. If you've not listened to these before, listen to the sinewave version *first*!


* [Ex. 1 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex1_sws.wav) | [Ex. 1 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex1.wav) | `-d 4 --high 300 --low 100 -o 4`
* [Ex. 2 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex2_sws.wav) | [Ex. 2 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex2.wav) | `-o 5 --window 200 --low 200`
* [Ex. 3 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex3_sws.wav) | [Ex. 3 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex3.wav) | `-o 4`
* [Ex. 4 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex4_sws.wav) | [Ex. 4 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex4.wav) | `-o 5 --high 2800 -d 8 --window 250`
* [Ex. 5 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex5_sws.wav) | [Ex. 5 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex5.wav) | `-d 12 --high 2000 --window 90`
* [Ex. 6 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex6_sws.wav) | [Ex. 6 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex6.wav) | `-d 8 --high 2500 --low 330 --window 90`
* [Ex. 7 (buzz)](http://johnhw.github.io/sinewave_speech/sounds/ex7_sws.wav) | [Ex. 7 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex7.wav) | `--buzz 80 --window 300 -d 8 --high 2000`
* [Ex. 8 (noise)](http://johnhw.github.io/sinewave_speech/sounds/ex8_sws.wav) | [Ex. 8 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex8.wav) | `--noise --low 200`
* [Ex. 9 (sinewave)](http://johnhw.github.io/sinewave_speech/sounds/ex9_sws.wav) | [Ex. 9 (original)](http://johnhw.github.io/sinewave_speech/sounds/ex9.wav) | `-d 2 --high 2800 --window 1500 -o 13`

*(these would probably be better if I spoke more clearly...)*

## Usage 

Requires `scipy` and `numpy` only.

Examples of use:

```sh
    python sws.py hello.wav    # converts hello.wav to hello_sws.wav

    # explicit output name
    python sws.py lpc.wav lpc_modified.wav 
```

More examples:

```sh
    # uses six sine wave components (order 6)
    python sws.py hello.wav -o 6 

    # sets the pre-filtering bandpass to be [40, 4000]Hz, and decimates 
    # by a factor of 4 before resynthesizing
    python sws.py hello.wav  --low 40 --high 4000 -d 4

    # uses modulation of a buzz (pulse train) @ 100Hz instead of sines
    python sws.py hello.wav --buzz 100

    # uses modulation of white noise instead of sinewaves
    python sws.py hello.wav --noise
```

## Command line parameters

        usage: sws.py [-h] [--low LOW] [--high HIGH] [--order ORDER] [--bw_amp BW_AMP]
                [--decimate DECIMATE] [--window WINDOW] [--interpolate] [--sine]
                [--buzz BUZZ] [--noise] [--overlap OVERLAP]
                input_wav [output_wav]

        positional arguments:
        input_wav             The input file, as a WAV file; ideally 44.1KHz mono.
        output_wav            The output file to write to; defaults to
                                <input>_sws.wav

        optional arguments:
        -h, --help            show this help message and exit
        --low LOW             Lowpass filter cutoff
        --high HIGH           Highpass filter cutoff
        --order ORDER, -o ORDER
                                Number of components in synthesis
        --bw_amp BW_AMP       Amplitude scaling by bandwidth; larger values flatten
                                amplitude; smaller values emphasise stronger formants
        --decimate DECIMATE, -d DECIMATE
                                Sample rate decimation before analysis
        --window WINDOW, -w WINDOW
                                LPC window size; smaller means faster changing signal;
                                larger is smoother
        --interpolate, -i     Enable interpolation
        --sine, -s            Resynthesise using sinewave speech (default)
        --buzz BUZZ, -b BUZZ  Resynthesie using buzz at given frequency (Hz)
        --noise, -n           Resynthesize using filtered white noise
        --overlap OVERLAP, -l OVERLAP
                                Window overlap, as fraction of the window length

## Technical details

Use 16 bit, 44.1KHz mono WAV files as input for best results.

### Transformation steps

* Input amplitude normalised 
* Bandpass filtered to [low, high] (order 4 Butterworth filter, non-causal)
* Pre-emphasis applied to slightly emphasise higher frequencies
* Decimated by a factor `d`
* Windowed into chunks of length `window`, overlapping by default by `window/2`
* Autocorrelation of signal computed
* RMS power of each window computed
* LPC computed from autocorrelation using Levinson-Durbin
* LPC converted to line spectral pairs (LSP)
* LSP converted to `order` (frequency, bandwidth) bands
* Window by window, each band resynthesied using sinewave oscillators , weighting amplitude by inverse (exponential) bandwidth
* Windows summed w/Hann window applied
* Overall amplitude envelope applied from estimated RMS power of each chunk
* Up-sampled by a factor of `d` and amplitude normalised

This implementation uses LPC estimation to estimate the formant centres using line spectral pairs to estimate formant frequencies and bandwidths. Amplitude of sinusoidal components is inversely proportional to bandwidth when resynthesis is performed, so noiser tracks become quieter.

Suggestions (especially pull requests!) on how to improve the quality of the output would be most welcome.

## Acknowledgements

* [Levinson-Durbin iteration by David Cournapeau](https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py) MIT Licensed
* ModFM synthesis algorithm used in buzz mode from *Lazzarini, V., & Timoney, J. (2010). New perspectives on distortion synthesis for virtual analog oscillators. Computer Music Journal, 34(1), 28-40.*
* The example sentences are read from [this list](https://www.cs.columbia.edu/~hgs/audio/harvard.html), originally from *IEEE Subcommittee on Subjective Measurements IEEE Recommended Practices for Speech Quality Measurements. IEEE Transactions on Audio and Electroacoustics. vol 17, 227-46, 1969*