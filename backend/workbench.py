import librosa
from dsp_core.audio_fx import reduce_volume
from dsp_core.visualizer import generate_comparison_plot

print("Loading test audio....")
y, sr = librosa.load("test_audio.wav",sr = None)

print("Applying DSP math...")
y_modified = reduce_volume(y, 0.5)

print("Drawing the graph...")
generate_comparison_plot(y,y_modified,sr,"workbench_test",start_sample=0,end_sample=100)

print("Done!")