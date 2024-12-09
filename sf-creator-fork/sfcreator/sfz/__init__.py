import os
from typing import List

from sfcreator.soundfont.soundfont import SoundFont, Sample


class SfzWriter:
    def write(self, directory: str, soundfont: SoundFont):
        f = open(directory + "/" + soundfont.instrument_name + ".sfz", "w")
        f.write("<group>\r\n")
        if soundfont.loop_mode is not None:
            f.write(f"loop_mode={soundfont.loop_mode}\r\n")
        if soundfont.polyphony is not None:
            f.write(f"loop_mode={soundfont.polyphony}\r\n")
        for root_key in soundfont.root_keys():
            self.write_root_key_sample_group(f, soundfont.samples_for_root_key(root_key))

    def write_root_key_sample_group(self, f, samples: List[Sample]):
        if len(samples) == 1:
            sample = samples[0]
            f.write(f"<region> sample={os.path.basename(sample.filename)}"
                    f" pitch_keycenter={str(sample.key_range.root_key)} lokey={str(sample.key_range.low_key)}"
                    f" hikey={str(sample.key_range.high_key)}\r\n")
        else:
            lorand = 0.0
            randstep = 1 / len(samples)
            for sample in samples:
                hirand = lorand + randstep
                f.write(f"<region> sample={os.path.basename(sample.filename)}"
                        f" pitch_keycenter={str(sample.key_range.root_key)} lokey={str(sample.key_range.low_key)}"
                        f" hikey={str(sample.key_range.high_key)}")
                if lorand > 0.0:
                    f.write(f" lorand={lorand}")
                if hirand < 1.0:
                    f.write(f" hirand={hirand}")
                f.write("\r\n")
                lorand = hirand
