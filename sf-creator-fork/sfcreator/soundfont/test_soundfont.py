from unittest import TestCase

from sfcreator.soundfont.soundfont import *


class TestHighLowKeyDistributor(TestCase):
    def test_distribution_of_samples_one_sample_only(self):
        distributor = HighLowKeyDistributor()

        sf = SoundFont(samples=[
            Sample("filename.wav", root_key=60)
        ])
        distributor.distribute(sf)

    def test_distribution_of_samples_unsorted_samples(self):
        distributor = HighLowKeyDistributor()
        sf = SoundFont(samples=[
            Sample("40.wav", root_key=40),
            Sample("60.wav", root_key=60),
            Sample("20.wav", root_key=20)
        ])
        distributor.distribute(sf)

        self.assertEqual(Sample("20.wav", root_key=20, low_key=20, high_key=30), sf.samples[0])
        self.assertEqual(Sample("40.wav", root_key=40, low_key=31, high_key=50), sf.samples[1])
        self.assertEqual(Sample("60.wav", root_key=60, low_key=51, high_key=108), sf.samples[2])

    def test_distribution_of_samples_over_108(self):
        distributor = HighLowKeyDistributor()
        sf = SoundFont(samples=[
            Sample("40.wav", root_key=40),
            Sample("110.wav", root_key=110)
        ])
        distributor.distribute(sf)

        self.assertEqual(Sample("40.wav", root_key=40, low_key=21, high_key=75), sf.samples[0])
        self.assertEqual(Sample("110.wav", root_key=110, low_key=76, high_key=110), sf.samples[1])

    def test_distribution_of_samples_with_duplicated_notes(self):
        distributor = HighLowKeyDistributor()
        sf = SoundFont(samples=[
            Sample("40.wav", root_key=40),
            Sample("40-1.wav", root_key=40),
            Sample("60.wav", root_key=60),
            Sample("20.wav", root_key=20)
        ])
        distributor.distribute(sf)

        self.assertEqual(Sample("20.wav", root_key=20, low_key=20, high_key=30), sf.samples[0])
        self.assertEqual(Sample("40.wav", root_key=40, low_key=31, high_key=50), sf.samples[1])
        self.assertEqual(Sample("40-1.wav", root_key=40, low_key=31, high_key=50), sf.samples[2])
        self.assertEqual(Sample("60.wav", root_key=60, low_key=51, high_key=108), sf.samples[3])


class TestSoundFont(TestCase):
    def test_creation_of_soundfont(self):
        sf = SoundFont(samples=[
            Sample("filename.wav", root_key=60)
        ])
        self.assertEqual(sf.samples[0].key_range.high_key, 60)
        self.assertEqual(sf.samples[0].key_range.low_key, 60)


class TestNoteNameMidiNumberMapper(TestCase):
    def test_map_note_name(self):
        mapper = NoteNameMidiNumberMapper()
        self.assertEqual(21, mapper.map("A0"))
        self.assertEqual(108, mapper.map("C8"))
        self.assertEqual(61, mapper.map("C#4"))
        self.assertEqual(60, mapper.map("C4"))
