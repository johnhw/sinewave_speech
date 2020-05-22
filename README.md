# Sinewave Speech
<img src="imgs/sws.png" width="100%">

A Python implementation of sinewave speech. Converts WAV files of human-speech to [sinewave speech](https://en.wikipedia.org/wiki/Sinewave_synthesis) using linear predictive coding (LPC). This is a "simplified" representation of the speech with a small number of frequency and amplitude modulated sine waves. It is surprising how much remains intelligible after this transformation.

Listen to some sounds. If you've not listened to these before, listen to the sinewave version *first*!


* [Ex. 1 (sinewave)](http://johnhw.github.io/sinewave_speech/ex1_sws.wav) | [Ex. 1 (original)](http://johnhw.github.io/sinewave_speech/ex1.wav) | `-d 4 --hp 3000 --lp 100 -o 8`
* [Ex. 2 (sinewave)](http://johnhw.github.io/sinewave_speech/ex2_sws.wav) | [Ex. 2 (original)](http://johnhw.github.io/sinewave_speech/ex2.wav) | `-o 6 --window 200 --use_lsp`
* [Ex. 3 (sinewave)](http://johnhw.github.io/sinewave_speech/ex3_sws.wav) | [Ex. 3 (original)](http://johnhw.github.io/sinewave_speech/ex3.wav) | `-o 8`
* [Ex. 4 (sinewave)](http://johnhw.github.io/sinewave_speech/ex4_sws.wav) | [Ex. 4 (original)](http://johnhw.github.io/sinewave_speech/ex4.wav) | `-o 10 --hp 2800 -d 8 --window 250`
* [Ex. 5 (sinewave)](http://johnhw.github.io/sinewave_speech/ex5_sws.wav) | [Ex. 5 (original)](http://johnhw.github.io/sinewave_speech/ex5.wav) | `-d 12 --hp 2000 --window 90`
* [Ex. 6 (sinewave)](http://johnhw.github.io/sinewave_speech/ex6_sws.wav) | [Ex. 6 (original)](http://johnhw.github.io/sinewave_speech/ex6.wav) | `-o 5 --use_lsp --lp 330 --hp 2500`
* [Ex. 7 (buzz)](http://johnhw.github.io/sinewave_speech/ex7_sws.wav) | [Ex. 7 (original)](http://johnhw.github.io/sinewave_speech/ex7.wav) | ` --buzz 80 --window 300 -d 8 --hp 2000`
* [Ex. 8 (noise)](http://johnhw.github.io/sinewave_speech/ex8_sws.wav) | [Ex. 8 (original)](http://johnhw.github.io/sinewave_speech/ex8.wav) | `--noise --lp 200`
* [Ex. 9 (sinewave)](http://johnhw.github.io/sinewave_speech/ex9_sws.wav) | [Ex. 9 (original)](http://johnhw.github.io/sinewave_speech/ex9.wav) | `--lp 100 --hp 2800 -d 2 --window 1500 -o 30`

*(these would probably be better if I spoke more clearly...)*

## Usage 

Requires `scipy` and `numpy` only.

Examples of use:

    python sws.py hello.wav    # converts hello.wav to hello_sws.wav

    # explicit output name
    python sws.py lpc.wav lpc_modified.wav 

More examples:

    # uses LSP mode, and six sine wave components (order 6)
    python sws.py hello.wav -o 6 --use-lsp 

    # sets the pre-filtering bandpass to be [40, 4000]Hz, and decimates 
    # by a factor of 8 before resynthesizing
    python sws.py hello.wav  --lp 40 --hp 4000 -d 8

    # uses modulation of a buzz (pulse train) @ 100Hz instead of sines
    python sws.py hello.wav --buzz 100

    # uses modulation of white noise instead of sinewaves
    python sws.py hello.wav --noise


## Command line parameters

    usage: sws.py [-h] [--lp LP] [--hp HP] [--order ORDER] [--use_lsp]
                [--decimate DECIMATE] [--window WINDOW] [--sine] [--buzz BUZZ]
                [--noise] input_wav [output_wav]

    positional arguments:
    input_wav             The input file, as a WAV file; ideally 44.1KHz mono.
    output_wav            The output file to write to; defaults to
                            <input>_sws.wav

    optional arguments:
    -h, --help            show this help message and exit
    --lp LP               Low frequency cutoff; removes frequencies below this
    --hp HP               High frequency cutoff; removes frequenceis above this
    --order ORDER, -o ORDER
                            LPC order; number of components in synthesis
    --use_lsp, -l         LPC order; number of components in synthesis
    --decimate DECIMATE, -d DECIMATE
                            Sample rate decimation before analysis
    --window WINDOW, -w WINDOW
                            LPC window size; smaller means faster changing signal;
                            larger is smoother
    --sine, -s            Resynthesise using sinewave speech (default)
    --buzz BUZZ, -b BUZZ  Resynthesie using buzz at given frequency (Hz)
    --noise, -n           Resynthesize using filtered white noise

## Technical details

This implementation uses LPC estimation to estimate the formant centres. This form can either be used directly or using line spectral pairs (theoretically more stable, but doesn't always sound good due to very even spacing of formants through the speech band).

Suggestions (especially pull requests!) on how to improve the quality of the output would be most welcome.

## Acknowledgements

* [Levinson-Durbin iteration by David Cournapeau](https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py) MIT Licensed
* ModFM synthesis algorithm used in buzz mode from *Lazzarini, V., & Timoney, J. (2010). New perspectives on distortion synthesis for virtual analog oscillators. Computer Music Journal, 34(1), 28-40.*
* The example sentences are read from [this list](https://www.cs.columbia.edu/~hgs/audio/harvard.html), originally from *IEEE Subcommittee on Subjective Measurements IEEE Recommended Practices for Speech Quality Measurements. IEEE Transactions on Audio and Electroacoustics. vol 17, 227-46, 1969*