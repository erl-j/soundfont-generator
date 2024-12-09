from typing import List


class KeyRange:
    def __init__(self, root_key: int = 60, low_key: int = None, high_key: int = None):
        self.root_key = root_key
        self.low_key = low_key
        self.high_key = high_key
        if self.low_key is None:
            self.low_key = self.root_key
        if self.high_key is None:
            self.high_key = self.root_key

    def in_range(self, key):
        return self.low_key <= key <= self.high_key

    def __eq__(self, other):
        return self.low_key == other.low_key and self.root_key == other.root_key and self.high_key == other.high_key


class VelocityRange:
    def __init__(self, low_velocity: int = 0, high_velocity: int = 127):
        self.low_velocity = low_velocity
        self.high_velocity = high_velocity

    def in_range(self, key):
        return self.low_velocity <= key <= self.high_velocity

    def __eq__(self, other):
        return self.low_velocity == other.low_velocity and self.high_velocity == other.root_key and\
               self.high_velocity == other.high_key


class Sample:
    def __init__(self, filename: str, root_key: int = 60, low_key: int = None, high_key: int = None):
        self.velocity_range = VelocityRange()
        self.filename = filename
        self.key_range = KeyRange(root_key, low_key, high_key)
        self.index = 0

    def __repr__(self) -> str:
        return 'Sample(filename=' + self.filename + ', root_key=' + str(self.key_range.root_key) + ', low_key=' + str(
            self.key_range.low_key) + ', high_key=' + str(self.key_range.high_key) + ')'

    def __eq__(self, other):
        return self.filename == other.filename and self.key_range == other.key_range and self.index == other.index


class SoundFont:
    def __init__(self, samples: List[Sample], loop_mode: str = "no_loop", polyphony: str = None, release: int = None,
                 instrument_name=None):
        self.instrument_name = instrument_name
        self.samples = samples
        self.loop_mode = loop_mode
        self.polyphony = polyphony
        self.release = release

    def root_keys(self):
        return sorted(set([sample.key_range.root_key for sample in self.samples]))

    def range_for_key(self, key) -> KeyRange:
        samples_in_range = [sample for sample in self.samples if sample.key_range.in_range(key)]
        return samples_in_range[0].key_range if len(samples_in_range) > 0 else None

    def samples_for_root_key(self, root_key):
        return [sample for sample in self.samples if sample.key_range.root_key == root_key]

    def set_range(self, root_key, low_key=None, high_key=None):
        for sample in self.samples_for_root_key(root_key):
            if low_key is not None:
                sample.key_range.low_key = low_key
            if high_key is not None:
                sample.key_range.high_key = high_key


class HighLowKeyDistributor:
    def distribute(self, soundfont: SoundFont, low_key: int = 21, high_key: int = 108):
        soundfont.samples.sort(key=lambda sample: sample.key_range.root_key, reverse=False)

        prev_root_key: int = None
        for root_key in soundfont.root_keys():
            range = soundfont.range_for_key(root_key)
            if prev_root_key is None:
                lo_key = min(low_key, range.low_key)
                soundfont.set_range(root_key, low_key=min(low_key, lo_key))
            else:
                prev_range = soundfont.range_for_key(prev_root_key)
                mid_sample_key = int((range.low_key - prev_range.high_key) / 2) + prev_range.high_key
                soundfont.set_range(prev_root_key, high_key=mid_sample_key)
                soundfont.set_range(root_key, low_key=mid_sample_key + 1)
            prev_root_key = root_key
        soundfont.set_range(soundfont.root_keys()[-1],
                            high_key=max(high_key, soundfont.range_for_key(soundfont.root_keys()[-1]).high_key))


class NoteNameMidiNumberMapper:
    def __init__(self):
        self.index_offset = 21
        self.note_name_midi_number_map: List[str] = []
        for octave_number in range(0, 8):
            for c in range(ord('A'), ord('B') + 1):
                self._add_note(c, octave_number)
            for c in range(ord('C'), ord('G') + 1):
                self._add_note(c, octave_number + 1)
        self.note_name_midi_number_map.append("C8")

    def _add_note(self, c, octave_number):
        self.note_name_midi_number_map.append(f"{chr(c)}{str(octave_number)}")
        if chr(c) != "A" and chr(c) != "E":
            self.note_name_midi_number_map.append(f"{chr(c)}#{str(octave_number)}")

    def mapped_sample(self, filename: str):
        for note in self.note_name_midi_number_map:
            if note in filename:
                return Sample(filename, self.map(note))
        return Sample(filename)

    def map(self, note_name: str):
        return self.note_name_midi_number_map.index(note_name) + self.index_offset
