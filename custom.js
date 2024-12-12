function previewPlayer() {
    class KeyboardPlayer {
        constructor(containerId) {
            this.container = document.getElementById(containerId);
            this.initializeProperties();
            this.loadToneJS().then(() => this.init());
            this.setupWavFileObserver();

            this.container.addEventListener('click', (e) => {
                e.stopPropagation();
                if (!this.keyboardEnabled) {
                    this.enableKeyboard();
                }
            });

            document.addEventListener('click', (e) => {
                if (!this.container.contains(e.target)) {
                    this.disableKeyboard();
                }
            });

            this.disableKeyboard();
        }

        initializeProperties() {
            this.sampler = null;
            this.keyboardEnabled = true;
            this.layout = null;
            this.rootPitch = 36;
            this.columnOffset = -12;
            this.rowOffset = 4;
            this.activeNotes = new Map();
            this.reverb = null;
            this.masterGain = null;
            this.releaseTime = 1;
            this.noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            this.majorScale = [0, 2, 4, 5, 7, 9, 11];
            this.isLoading = false;
        }

        enableKeyboard() {
            if (this.isLoading) return;
            this.keyboardEnabled = true;
            this.container.style.opacity = '1';
            this.setLoaderState('inactive');
        }

        disableKeyboard() {
            this.keyboardEnabled = false;
            this.container.style.opacity = '0.5';
        }

        setLoaderState(state) {
            const loader = this.container.querySelector('.loader');
            loader.className = `loader ${state}`;
            loader.style.display = state === 'active' ? 'flex' : 'none';

            if (state === 'active') {
                loader.innerHTML = 'Loading samples... <span class="loader-emoji">🎹</span>';
            } else {
                loader.innerHTML = 'Ready <span class="loader-emoji">✨</span>';
            }
        }

        async loadToneJS() {
            if (window.Tone) return;
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js';
            return new Promise((resolve, reject) => {
                script.onload = resolve;
                script.onerror = () => reject(new Error('Failed to load Tone.js'));
                document.head.appendChild(script);
            });
        }

        init() {
            this.createUI();
            this.detectKeyboardLayout();
            this.setupEventListeners();
            this.initializeEffects();
            this.initializeSampler();
        }

        createUI() {
            this.container.innerHTML = `
                <div class="controls-section">
                    <details>
                        <summary><h3>Master & Effects</h3></summary>
                        <div class="control-group">
                            <label class="slider-label">Release: <span class="release-value">1s</span></label>
                            <input type="range" class="control-slider release-slider" min="0" max="3" step="0.1" value="0.1">
                        </div>
                        <div class="controls-row">
                            <div class="control-group half-width">
                                <label class="slider-label">Reverb Mix: <span class="reverb-mix-value">20%</span></label>
                                <input type="range" class="control-slider reverb-mix-slider" min="0" max="100" value="20">
                            </div>
                            <div class="control-group half-width">
                                <label class="slider-label">Master: <span class="master-value">100%</span></label>
                                <input type="range" class="control-slider master-slider" min="0" max="200" value="100">
                            </div>
                        </div>
                    </details>
                </div>
                <div class="keyboard"></div>
                <div class="controls-section">
                    <details>
                        <summary><h3>Keyboard Mapping</h3></summary>
                        <div class="control-group">
                            <label class="slider-label">Root Pitch: <span class="root-value">C4</span></label>
                            <input type="range" class="control-slider root-slider" min="24" max="84" value="60">
                        </div>
                        <div class="controls-row">
                            <div class="control-group half-width">
                                <label class="slider-label">Column Offset: <span class="column-value">-12</span> keys</label>
                                <input type="range" class="control-slider column-slider" min="-20" max="20" value="2">
                            </div>
                            <div class="control-group half-width">
                                <label class="slider-label">Row Offset: <span class="row-value">4</span> degrees</label>
                                <input type="range" class="control-slider row-slider" min="1" max="20" value="4">
                            </div>
                        </div>
                    </details>
                </div>
                <div class="loader inactive">
                    Ready <span class="loader-emoji">🦦</span>
                </div>
            `;
            this.cacheElements();
        }

        cacheElements() {
            const selectors = {
                keyboard: '.keyboard',
                masterSlider: '.master-slider',
                masterValue: '.master-value',
                rootSlider: '.root-slider',
                rootValue: '.root-value',
                columnSlider: '.column-slider',
                columnValue: '.column-value',
                rowSlider: '.row-slider',
                rowValue: '.row-value',
                releaseSlider: '.release-slider',
                releaseValue: '.release-value',
                reverbMixSlider: '.reverb-mix-slider',
                reverbMixValue: '.reverb-mix-value'
            };
            this.elements = Object.fromEntries(
                Object.entries(selectors).map(([key, selector]) =>
                    [key, this.container.querySelector(selector)]
                )
            );
        }

        setupEventListeners() {
            const handlers = {
                masterSlider: e => {
                    const gain = parseInt(e.target.value) / 100;
                    this.masterGain.gain.value = gain;
                    this.elements.masterValue.textContent = `${e.target.value}%`;
                },
                releaseSlider: e => {
                    this.releaseTime = parseFloat(e.target.value);
                    this.elements.releaseValue.textContent = `${this.releaseTime}s`;
                },
                reverbMixSlider: e => {
                    const wetness = parseInt(e.target.value) / 100;
                    this.reverb.wet.value = wetness;
                    this.elements.reverbMixValue.textContent = `${e.target.value}%`;
                },
                rootSlider: e => {
                    this.rootPitch = parseInt(e.target.value);
                    this.elements.rootValue.textContent = this.midiToNoteName(this.rootPitch);
                    this.updateNotes();
                },
                columnSlider: e => {
                    this.columnOffset = parseInt(e.target.value);
                    this.elements.columnValue.textContent = this.columnOffset;
                    this.updateNotes();
                },
                rowSlider: e => {
                    this.rowOffset = parseInt(e.target.value);
                    this.elements.rowValue.textContent = this.rowOffset;
                    this.updateNotes();
                }
            };

            Object.entries(handlers).forEach(([element, handler]) =>
                this.elements[element].addEventListener('input', handler));

            document.addEventListener('mouseup', () => this.handleMouseUp());
            document.addEventListener('keydown', e => !e.repeat && this.handleKeyEvent(e, true));
            document.addEventListener('keyup', e => this.handleKeyEvent(e, false));
        }

        setupWavFileObserver() {
            const observer = new MutationObserver((mutations) => {
                const hasDownloadLinkChanges = mutations.some(mutation =>
                    mutation.type === 'childList' &&
                    mutation.target.classList.contains('download-link')
                );

                if (hasDownloadLinkChanges) {
                    this.initializeSampler();
                    this.enableKeyboard();
                    const keyboardTop = this.container.querySelector('.keyboard').getBoundingClientRect().top;
                    window.scrollTo(0, keyboardTop - window.innerHeight / 2, { behavior: 'smooth' });
                }
            });

            const wavFilesContainer = document.getElementById('individual-wav-files');
            if (wavFilesContainer) {
                observer.observe(wavFilesContainer, {
                    childList: true,
                    subtree: true
                });
            }
        }

        async initializeSampler() {
            this.isLoading = true;
            this.setLoaderState('active');
            this.container.querySelectorAll('.key').forEach(key => key.style.opacity = '0.5');

            const requiredNotes = ['C1', 'F#1', 'C2', 'F#2', 'C3', 'F#3', 'C4', 'F#4', 'C5', 'F#5'];
            const urls = {};
            let allNotesFound = true;

            for (const note of requiredNotes) {
                // Look for download links with the note name in them
                const downloadLink = document.querySelector(`.download-link[href*="${note}.wav"], .download-link[href*="${note.replace('#', '')}.wav"]`);
                if (downloadLink?.href) {
                    urls[note] = downloadLink.href;
                } else {
                    console.log(`Could not find sample for note: ${note}`);
                    allNotesFound = false;
                    break;
                }
            }

            if (!allNotesFound) {
                this.handleSamplerError();
                return;
            }

            if (this.sampler) {
                this.sampler.dispose();
            }

            try {
                this.sampler = new Tone.Sampler({
                    urls,
                    onload: () => this.handleSamplerLoad(),
                    onerror: (error) => {
                        console.error('Sampler loading error:', error);
                        this.handleSamplerError();
                    }
                }).connect(this.reverb);
            } catch (error) {
                console.error('Sampler initialization error:', error);
                this.handleSamplerError();
            }
        }

        handleSamplerLoad() {
            console.log('All samples loaded successfully');
            this.isLoading = false;
            this.container.querySelectorAll('.key').forEach(key => key.style.opacity = '1');
            this.setLoaderState('inactive');
            this.enableKeyboard();
        }

        handleSamplerError() {
            console.log('No WAV files found');
            this.isLoading = false;
            this.setLoaderState('inactive');
        }

        initializeEffects() {
            this.masterGain = new Tone.Gain(1).toDestination();
            this.reverb = new Tone.Reverb({
                decay: 1.5,
                wet: 0.5,
                preDelay: 0.01
            }).connect(this.masterGain);
        }

        detectKeyboardLayout() {
            this.layout = {
                keys: [
                    { keys: '1234567890'.split(''), offset: 0 },
                    { keys: 'QWERTYUIOP'.split(''), offset: 1 },
                    { keys: 'ASDFGHJKL'.split(''), offset: 1.5 },
                    { keys: 'ZXCVBNM,.'.split(''), offset: 2 }
                ]
            }.keys;
            this.createKeyboard();
        }

        createKeyboard() {
            this.elements.keyboard.innerHTML = '';
            this.layout.forEach((row, rowIndex) => {
                const rowElement = document.createElement('div');
                rowElement.className = 'keyboard-row';
                rowElement.style.paddingLeft = `${row.offset * 3}%`;
                row.keys.forEach(key => rowElement.appendChild(this.createKey(key)));
                this.elements.keyboard.appendChild(rowElement);
            });
            this.updateNotes();
        }

        createKey(keyLabel) {
            const key = document.createElement('div');
            key.className = 'key';
            key.innerHTML = `
                <div class="key-label">${keyLabel}</div>
                <div class="note-label"></div>
            `;
            key.addEventListener('mousedown', () => this.startNote(key));
            key.addEventListener('mouseenter', e => e.buttons === 1 && this.startNote(key));
            key.addEventListener('mouseleave', () => this.stopNote(key));
            return key;
        }

        updateNotes() {
            Array.from(this.elements.keyboard.children).forEach((row, rowIndex) => {
                Array.from(row.children).forEach((key, columnIndex) => {
                    const horizontalDistance = columnIndex - this.columnOffset;
                    const verticalDistance = rowIndex * this.rowOffset;
                    const totalScaleDegrees = horizontalDistance - verticalDistance;
                    const octaves = Math.floor(totalScaleDegrees / 7);
                    const remainingDegrees = ((totalScaleDegrees % 7) + 7) % 7;
                    const semitonesFromRoot = this.majorScale[remainingDegrees] + (octaves * 12);
                    const midiNote = this.rootPitch + semitonesFromRoot;

                    this.updateKeyDisplay(key, midiNote);
                });
            });
        }

        updateKeyDisplay(key, midiNote) {
            const isBaseRoot = midiNote === this.rootPitch;
            const isOctaveRoot = midiNote % 12 === this.rootPitch % 12;
            key.style.backgroundColor = isBaseRoot ? '#90EE90' : isOctaveRoot ? '#E8F5E9' : '';
            const noteName = this.midiToNoteName(midiNote);
            key.querySelector('.note-label').textContent = noteName;
            key.dataset.note = noteName;
            key.dataset.midi = midiNote;
        }

        handleKeyEvent(e, isKeyDown) {
            if (!this.keyboardEnabled || !this.sampler) return;
            const keyElement = this.findKeyElement(e.key.toUpperCase());
            if (keyElement) {
                e.preventDefault();
                isKeyDown ? this.startNote(keyElement) : this.stopNote(keyElement);
            }
        }

        startNote(keyElement) {
            if (!this.sampler || !keyElement || this.activeNotes.has(keyElement)) return;
            const note = keyElement.dataset.note;
            if (!note) return;

            Tone.start().then(() => {
                this.sampler.triggerAttack(note);
                this.activeNotes.set(keyElement, { note });
                this.animateKey(keyElement, true);
            });
        }

        stopNote(keyElement) {
            if (!this.sampler || !keyElement) return;
            const noteInfo = this.activeNotes.get(keyElement);
            if (noteInfo) {
                this.sampler.triggerRelease(noteInfo.note, "+" + this.releaseTime);
                this.activeNotes.delete(keyElement);
                this.animateKey(keyElement, false);
            }
        }

        handleMouseUp() {
            this.activeNotes.forEach((_, keyElement) => this.stopNote(keyElement));
        }

        findKeyElement(keyLabel) {
            for (const row of this.elements.keyboard.children) {
                for (const key of row.children) {
                    if (key.querySelector('.key-label').textContent === keyLabel) return key;
                }
            }
            return null;
        }

        animateKey(keyElement, isDown) {
            const midiNote = parseInt(keyElement.dataset.midi);
            const isBaseRoot = midiNote === this.rootPitch;
            const isOctaveRoot = midiNote % 12 === this.rootPitch % 12;

            keyElement.style.transform = isDown ? 'scale(0.95)' : '';
            keyElement.style.backgroundColor = isBaseRoot ? '#90EE90' :
                isOctaveRoot ? '#E8F5E9' :
                    isDown ? '#f0f0f0' : '';
        }

        midiToNoteName(midiNumber) {
            const octave = Math.floor(midiNumber / 12) - 1;
            return `${this.noteNames[midiNumber % 12]}${octave}`;
        }
    }

    let container = document.getElementById('keyboard-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'keyboard-container';
        document.body.appendChild(container);
    }
    new KeyboardPlayer('keyboard-container');
}