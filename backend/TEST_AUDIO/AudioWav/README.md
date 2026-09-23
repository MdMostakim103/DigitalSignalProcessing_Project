# DSP Studio — test audio

Files for hearing what each module actually does. The speech files are cut from
a real 33 s voice recording (`../AudioWav/voice.wav`); the tones, sweeps, drums
and impulse response are synthesized so their content is exactly known.

All files: 44.1 kHz, mono, 16-bit PCM.

---

## Module 1 — Amplitude & Dynamics

| File | Do this | What you should hear / see |
|---|---|---|
| `speech_quiet.wav` | Gain `1×` → `4×` | Peaks at 0.072 (−23 dB), so it starts almost inaudible. At 4× it becomes normal-ish loudness. |
| `speech_clean.wav` | Gain `4×` | Already peaks at 0.9, so 4× drives it well past full scale → audible **clipping distortion**, harsh and buzzy. |

> Known issue: the graph plots the unclipped float array while the WAV clips at
> ±1, so at high gain the picture looks clean while the audio is distorted.

## Module 2 — Frequency & Filtering

| File | Do this | What you should hear |
|---|---|---|
| `bass_100Hz_plus_treble_6kHz.wav` | Lowpass @ 1000 Hz | The bright 6 kHz tone vanishes, only the low hum remains (measured −123 dB). |
| `bass_100Hz_plus_treble_6kHz.wav` | Highpass @ 1000 Hz | The opposite: bass disappears, only the thin high tone (−151 dB on the bass). |
| `sweep_20Hz_18kHz.wav` | Lowpass @ 2000 Hz | The rising sweep climbs, then **fades out** as it passes the cutoff. The clearest way to *hear* a cutoff frequency. |
| `sweep_20Hz_18kHz.wav` | Bandpass 500–2000 Hz | Sweep is silent, becomes audible in the middle, goes silent again. |
| `voice_with_50Hz_hum.wav` | Bandstop 40–60 Hz | The steady low buzz disappears, voice untouched (−99 dB at 50 Hz). |
| `speech_with_fan_noise.wav` | Highpass @ 300 Hz | Fan rumble drops noticeably (−13 dB below 300 Hz) while speech stays intact. Real noise reduction via filtering. |

Also worth doing: same cutoff, **order 1 vs order 10** — hear how much sharper
the transition gets. And compare **Butterworth vs Elliptic** at the same order:
elliptic cuts harder but adds ripple.

## Module 3 — Time-Domain Processing

| File | Do this | What you should hear |
|---|---|---|
| `drum_hits.wav` | Echo / Delay | Sharp transients make the repeats unmistakable — you can count them. Far clearer than using speech. |
| `drum_hits.wav` + `room_impulse_response.wav` | Convolution (IR = the room file) | Dry clicks become a drum in a room. This is convolution reverb. |
| `speech_clean.wav` + `room_impulse_response.wav` | Convolution | Voice sounds recorded in a room instead of close-mic'd. |

## Module 4 — Speech & Activity

| File | Do this | What you should see |
|---|---|---|
| `speech_with_pauses.wav` | Threshold 15% | ~56% of frames marked active. Clear green/grey alternation matching the speech bursts. This is the file to demo with. |
| `speech_with_fan_noise.wav` | Threshold 15% | **100% active** — the fan never drops below threshold, so naive energy VAD fails completely. Raise the threshold to ~40% to recover the pauses. A genuinely instructive failure. |

## Module 5 — Spectral Detection

| File | Expected detection |
|---|---|
| `tone_A4_440Hz.wav` | 440.0 Hz → **A4**, in tune |
| `tone_C4_262Hz.wav` | 261.6 Hz → **C4** |
| `tone_E2_82Hz.wav` | 82.4 Hz → **E2** (set search range min below 80 Hz) |
| `speech_clean.wav` | Whatever the speaker's pitch is — messier, since speech is not one steady tone. |

All three tones contain harmonics, not pure sines, so the detector has to pick
the fundamental rather than the loudest partial.

## Module 6 — Voice Morphing

| File | Do this | What you should hear |
|---|---|---|
| `speech_clean.wav` | Pitch shift `+7` | Same words, same speed, higher voice. |
| `speech_clean.wav` | Pitch shift `−7` | Deeper voice, timing unchanged. |
| `speech_clean.wav` | Time stretch `0.6×` | Slower, longer — **pitch does not drop**. Contrast with tape/varispeed. |
| `speech_clean.wav` | Robot | Flat monotone buzz. Magnitude spectrum is *identical* — all of that change came from zeroing the phase. |
| `speech_clean.wav` | Whisper | Breathy, unvoiced. Again, same magnitudes, random phase. |

Robot and Whisper are the payoff of the whole project: identical `|X(f)|`,
completely different sound.

---

## Not currently testable in the UI

`apply_noise_reduction` (spectral gating) exists in the backend but no page
calls it — `AudioInput.jsx` is the only component that requests
`effect="noise"`, and it is never imported. Use Module 2's highpass on
`speech_with_fan_noise.wav` for a filtering-based equivalent, or call the API
directly.
