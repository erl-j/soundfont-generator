# Soundfont creation

This library aims to create a soundfont based on a directory containing sound files (`.wav`).

Current support:

* [SFZ Format](https://sfzformat.com/)
* [DecentSampler](https://www.decentsamples.com/product/decent-sampler-plugin/)

Planned support:

* [Soundfont 2 `sf2`](https://en.wikipedia.org/wiki/SoundFont) (This will be more tricky, as it is a binary format,
as [Polyphone](https://www.polyphone-soundfonts.com/) is able to convert `sfz` to `sf2`, I will postpone this)

## Project setup

### Creating a project specific virtual environment (recommended)

You can omit this step if you are ok with installing the dependencies
system wide and go directly to the next step: [Installing dependencies](#installing-dependencies).

```
virtualenv venv
source venv/bin/activate
```
or, under Windows:
```
virtualenv venv
venv\Scripts\activate
```

### Installing dependencies

```
pip install -r requirements.txt
```

## Run

This will create a file `soundfont.sfz` alongside the `wav` files in the given directory.

```
python main.py sfz <directory-to-wave-files>
```

Run `python main.py --help` or `python main.py <command> --help`, where `<command>` can be `sfz` or `decentsampler` for now,
to get the full list of arguments.

## Automatic note detection and mapping of samples

The given samples are scanned for note names (A0 to C8). If a note name is found in a filename of a sample, the midi
note for this sample will be set automatically.

In addition to that, in case of missing samples in between for certain notes an automatic distribution is calculated, so that all notes between A0 and C8 are covered.

If there are two samples for the same note available, a round robin/random change is assumed.

## TODO and resources

- [ ] Make automatic distribution over all midi notes from 21 to 108 optional, and add an option to configure the highest and the lowest, ideally relative to the hightest and lowest pitch of the samples
- [ ] Detect pitch automatically (for melodic instruments at least), using https://pypi.org/project/crepe/

### More SFZ Support

The best starting point for SFZ is https://sfzformat.com/.

- [ ] Look at [SFZ Python Automapper by Peter Eastman](https://vis.versilstudios.com/sfzconverter.html#u13452-4), this looks like there is a lot that can be reused for sfz files
- [ ] And https://github.com/freepats/freepats-tools, too?
- [ ] Add support for velocity levels
- [ ] Add support for more options supported by SFZ, reverb, effects, attack, release and so on

### DecentSampler support

An XML based format developed by David Hilowitz (see https://youtu.be/UxPRmD_RNCY).

- [x] Create an XML Schema for highlighting and autocompletion
- [x] Implement `DecentSamplerWriter`
- [x] Add options for UI (cover)
- [ ] ...and effects

### SF2 Support

This will get tricky, as this is a binary format with not too much examples. There are a few applications reading or even writing sf2 out there, at least at a very basic level.
But as [Polyphone](https://www.polyphone-soundfonts.com/) is able to convert `sfz` to `sf2`, I will postpone this.

* Basic SFZ to SF2 converter in python: https://github.com/freepats/freepats-tools
* C++ library [sf2cute](http://gocha.github.io/sf2cute/)
* Python library reading sf2:  https://pypi.org/project/sf2utils/
* C# code writing basic sf2 file: https://github.com/Kermalis/SoundFont2
* Code of Polyphone: https://github.com/davy7125/polyphone
