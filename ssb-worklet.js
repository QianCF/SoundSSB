/* 对讲机 AudioWorklet：固定 CH2 (18–26 kHz) */
const CH_LOW = 15000;
const CH_HIGH = 23000;
const VOICE_HIGH = 8000;
const FFT_SIZE = 1024;
const HOP_SIZE = FFT_SIZE >> 2;

function complexFFT(re, im, inverse) {
    const n = re.length;
    const sign = inverse ? 1 : -1;
    const scale = inverse ? 1 / n : 1;
    for (let i = 0, j = 0; i < n; i++) {
        if (i < j) {
            const tr = re[i]; re[i] = re[j]; re[j] = tr;
            const ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        let bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
    }
    for (let len = 2; len <= n; len <<= 1) {
        const half = len >> 1;
        const angle = (sign * 2 * Math.PI) / len;
        const wlenRe = Math.cos(angle);
        const wlenIm = Math.sin(angle);
        for (let i = 0; i < n; i += len) {
            let wRe = 1, wIm = 0;
            for (let j = 0; j < half; j++) {
                const uRe = re[i + j], uIm = im[i + j];
                const tRe = re[i + j + half] * wRe - im[i + j + half] * wIm;
                const tIm = re[i + j + half] * wIm + im[i + j + half] * wRe;
                re[i + j] = uRe + tRe;
                im[i + j] = uIm + tIm;
                re[i + j + half] = uRe - tRe;
                im[i + j + half] = uIm - tIm;
                const nwr = wRe * wlenRe - wIm * wlenIm;
                const nwi = wRe * wlenIm + wIm * wlenRe;
                wRe = nwr;
                wIm = nwi;
            }
        }
    }
    if (scale !== 1) {
        for (let i = 0; i < n; i++) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

function fft(realSamples) {
    const n = realSamples.length;
    const re = new Float32Array(realSamples);
    const im = new Float32Array(n);
    complexFFT(re, im, false);
    const magnitude = new Float32Array(n);
    const phase = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        magnitude[i] = Math.hypot(re[i], im[i]);
        phase[i] = Math.atan2(im[i], re[i]);
    }
    return { magnitude, phase };
}

function ifft(magnitude, phase) {
    const n = magnitude.length;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        re[i] = magnitude[i] * Math.cos(phase[i]);
        im[i] = magnitude[i] * Math.sin(phase[i]);
    }
    complexFFT(re, im, true);
    return re;
}

function shiftSpectrum(magnitude, phase, shiftBins, fftSize) {
    const newMagnitude = new Float32Array(fftSize);
    const newPhase = new Float32Array(fftSize);
    if (shiftBins > 0) {
        for (let i = 0; i < fftSize - shiftBins; i++) {
            newMagnitude[i + shiftBins] = magnitude[i];
            newPhase[i + shiftBins] = phase[i];
        }
    } else {
        const absShift = Math.abs(shiftBins);
        for (let i = absShift; i < fftSize; i++) {
            newMagnitude[i - absShift] = magnitude[i];
            newPhase[i - absShift] = phase[i];
        }
    }
    return { magnitude: newMagnitude, phase: newPhase };
}

class Fm2WalkieProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.transmitting = false;
        this.hannWindow = new Float32Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i++) {
            this.hannWindow[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
        }
        this.resetBuffers();

        this.port.onmessage = (e) => {
            const msg = e.data;
            if (!msg || !msg.type) return;
            if (msg.type === 'transmit') {
                this.transmitting = !!msg.value;
            } else if (msg.type === 'reset') {
                this.resetBuffers();
            }
        };
    }

    resetBuffers() {
        this.inputRing = new Float32Array(FFT_SIZE * 4);
        this.absSample = 0;
        this.outputReadIndex = 0;
        const olaLen = Math.max(FFT_SIZE * 32, Math.ceil(sampleRate * 15));
        this.ola = new Float32Array(olaLen);
    }

    ringIdx(i) {
        const len = this.inputRing.length;
        return ((i % len) + len) % len;
    }

    olaIdx(i) {
        const len = this.ola.length;
        return ((i % len) + len) % len;
    }

    processWindow(windowData, transmit) {
        const sr = sampleRate;
        const binHz = sr / FFT_SIZE;
        const { magnitude, phase } = fft(windowData);

        const keepLow = transmit ? 0 : CH_LOW;
        const keepHigh = transmit ? VOICE_HIGH : CH_HIGH;
        const shiftHz = transmit ? CH_LOW : -CH_LOW;
        const shiftBins = Math.round((shiftHz * FFT_SIZE) / sr);

        const keepLowBin = Math.max(0, Math.floor(keepLow / binHz));
        const keepHighBin = Math.min(Math.ceil(keepHigh / binHz), FFT_SIZE - 1);

        const maskedMag = new Float32Array(FFT_SIZE);
        const maskedPhase = new Float32Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i++) {
            if (i >= keepLowBin && i <= keepHighBin) {
                maskedMag[i] = magnitude[i];
                maskedPhase[i] = phase[i];
            }
        }

        const shifted = shiftSpectrum(maskedMag, maskedPhase, shiftBins, FFT_SIZE);
        const real = ifft(shifted.magnitude, shifted.phase);

        const synth = new Float32Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i++) {
            synth[i] = real[i] * this.hannWindow[i];
        }
        return synth;
    }

    processFrameAt(frameStart) {
        const windowData = new Float32Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i++) {
            windowData[i] = this.inputRing[this.ringIdx(frameStart + i)] * this.hannWindow[i];
        }
        const synth = this.processWindow(windowData, this.transmitting);
        for (let i = 0; i < FFT_SIZE; i++) {
            const idx = this.olaIdx(frameStart + i);
            this.ola[idx] += synth[i];
        }
    }

    pushSample(s) {
        this.inputRing[this.ringIdx(this.absSample)] = s;
        this.absSample++;
        if (this.absSample >= FFT_SIZE && ((this.absSample - FFT_SIZE) % HOP_SIZE === 0)) {
            this.processFrameAt(this.absSample - FFT_SIZE);
        }
    }

    pullSample() {
        if (this.outputReadIndex >= FFT_SIZE) {
            const playIdx = this.outputReadIndex - FFT_SIZE;
            const idx = this.olaIdx(playIdx);
            let v = this.ola[idx];
            this.ola[idx] = 0;
            if (v > 1) v = 1;
            else if (v < -1) v = -1;
            this.outputReadIndex++;
            return v;
        }
        this.outputReadIndex++;
        return 0;
    }

    process(inputs, outputs) {
        const input = inputs[0] && inputs[0][0];
        const output = outputs[0] && outputs[0][0];
        if (!output) return true;

        const len = output.length;
        if (!input) {
            for (let i = 0; i < len; i++) {
                this.pushSample(0);
                output[i] = this.pullSample();
            }
            return true;
        }

        for (let i = 0; i < len; i++) {
            this.pushSample(input[i]);
            output[i] = this.pullSample();
        }
        return true;
    }
}

registerProcessor('fm2-walkie-processor', Fm2WalkieProcessor);
