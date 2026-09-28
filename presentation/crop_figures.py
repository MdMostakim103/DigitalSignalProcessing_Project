from PIL import Image, ImageOps
import os

SRC = "D:/2-2/cse - 220/DSP_Project_Picture"
DST = "D:/2-2/cse - 220/DSP_Poject/presentation/figures"
os.makedirs(DST, exist_ok=True)

# (source_file, output_name, crop_box_or_None)
# crop_box = (left, top, right, bottom) in source pixel coordinates
JOBS = [
    ("amplitude_time_domain.png",        "amp_io.png",        (0, 520, 1326, 1200)),
    ("convolution_visualization.png",    "conv_viz.png",       None),
    ("echo_output.png",                  "echo_io.png",        None),
    ("delay_output.png",                 "delay_io.png",       None),
    ("pitch_detector_output_frequency.png", "pitch_spectra.png", (0, 845, 1339, 1227)),
    ("quantization_output.png",          "quant_mid.png",      (0, 395, 808, 878)),
    ("sampling_output.png",              "sample_mid.png",     (0, 420, 843, 878)),
    ("filtering.png",                    "filter_response.png",(0, 515, 913, 750)),
    ("voice_morphing.png",               "morph_panels.png",   (0, 750, 1174, 1300)),
    ("spectogram_input.png",             "spectrogram_mask.png",(0, 368, 1345, 900)),
    ("spectogram_output.png",            "spectrogram_before_after.png", None),
]

ROUND_RADIUS = 14

def round_corners(im, radius):
    mask = Image.new("L", im.size, 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, im.size[0], im.size[1]], radius=radius, fill=255)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    return out

for src_name, out_name, box in JOBS:
    im = Image.open(os.path.join(SRC, src_name)).convert("RGB")
    if box is not None:
        im = im.crop(box)
    im = round_corners(im, ROUND_RADIUS)
    out_path = os.path.join(DST, out_name)
    im.save(out_path)
    print(f"{src_name} -> {out_name}  {im.size}")

print("Done.")
