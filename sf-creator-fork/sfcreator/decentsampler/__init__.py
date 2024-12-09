import os
from typing import List
from lxml import etree
from sfcreator.soundfont.soundfont import SoundFont, Sample

class DecentSamplerWriter:
    def write(self, directory: str, soundfont: SoundFont, image: str):
        root = etree.Element("DecentSampler", pluginVersion="1")

        if image is not None:
            ui = etree.Element("ui", bgImage=image, width="812", height="375", layoutMode="relative", bgMode="top_left")
            root.append(ui)
            tab = etree.Element("tab", name="main")
            ui.append(tab)

        groups = etree.Element("groups")
        root.append(groups)

        if soundfont.loop_mode == "one_shot":
            # assuming no sample will be longer than 20s
            groups.set("release", "20")
        elif soundfont.release is not None:
            groups.set("release", str(soundfont.release))

        for root_key in soundfont.root_keys():
            self.write_root_key_sample_group(groups, soundfont.samples_for_root_key(root_key))

        filename = directory + "/" + soundfont.instrument_name + ".dspreset"
        print("Writing to " + filename)
        et = etree.ElementTree(root)
        et.write(filename, pretty_print=True, encoding='utf-8', xml_declaration=True)

    def write_root_key_sample_group(self, groups, samples: List[Sample]):
        group = etree.Element("group")
        groups.append(group)
        if len(samples) > 1:
            group.set("seqMode", "random")
        for index, sample in enumerate(samples, start=1):
            xml_sample = etree.Element("sample")
            group.append(xml_sample)
            xml_sample.set("rootNote", str(sample.key_range.root_key))
            xml_sample.set("loNote", str(sample.key_range.low_key))
            xml_sample.set("hiNote", str(sample.key_range.high_key))
            xml_sample.set("loVel", str(sample.velocity_range.low_velocity))
            xml_sample.set("hiVel", str(sample.velocity_range.high_velocity))
            xml_sample.set("seqPosition", str(index))
            xml_sample.set("path", str(os.path.basename(sample.filename)))
