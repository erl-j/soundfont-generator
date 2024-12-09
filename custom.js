function previewPlayer() {
    class KeyboardPlayer {
        constructor(containerId) {
            this.container = document.getElementById(containerId);
            this.initializeProperties();
            this.loadToneJS().then(() => this.init());
            this.setupWavFileObserver();
        

            // Add click handlers for activation/deactivation
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

            // disable keyboard
            this.disableKeyboard();
        }

        enableKeyboard() {
            this.keyboardEnabled = true;
            this.container.style.opacity = '1';
        }

        disableKeyboard() {
            this.keyboardEnabled = false;
            this.container.style.opacity = '0.5';
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
                    // scroll so middle of keyboard is in centre of viewport
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

        initializeProperties() {
            this.sampler = null;
            this.keyboardEnabled = true;
            this.layout = null;
            this.rootPitch = 60;
            this.columnOffset = 2;
            this.rowOffset = 4;
            this.activeNotes = new Map();
            this.reverb = null;
            this.releaseTime = 0.1;
            this.noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            this.majorScale = [0, 2, 4, 5, 7, 9, 11];
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
                <div class="keyboard-container">
                    <div class="effects-controls">
                        <h3>Release & Reverb</h3>
                        <div class="effect-slider">
                            <label>Release: <span class="release-value">0.1s</span></label>
                            <input type="range" class="release-slider" min="0" max="3" step="0.1" value="0.1">
                        </div>
                        <div class="effect-slider">
                            <label>Reverb: <span class="reverb-value">50%</span></label>
                            <input type="range" class="reverb-slider" min="0" max="100" value="50">
                        </div>
                    </div>
                    <div class="keyboard"></div>
                    <br>
                      <div class="mapping-controls">
                        <h3>Keyboard Mapping</h3>
                        <div class="control-group">
                            <label>Root Pitch: <span class="root-value">C4</span></label>
                            <input type="range" class="root-slider" min="24" max="84" value="60">
                        </div>
                        <div class="control-group">
                            <label>Column Offset: <span class="column-value">2</span> keys from left</label>
                            <input type="range" class="column-slider" min="0" max="6" value="2">
                        </div>
                        <div class="control-group">
                            <label>Row Offset: <span class="row-value">4</span> scale degree(s)</label>
                            <input type="range" class="row-slider" min="1" max="20" value="4">
                        </div>
                    </div>
                </div>
            `;
            this.cacheElements();
        }

        cacheElements() {
            const selectors = {
                keyboard: '.keyboard',
                rootSlider: '.root-slider',
                rootValue: '.root-value',
                columnSlider: '.column-slider',
                columnValue: '.column-value',
                rowSlider: '.row-slider',
                rowValue: '.row-value',
                releaseSlider: '.release-slider',
                releaseValue: '.release-value',
                reverbSlider: '.reverb-slider',
                reverbValue: '.reverb-value'
            };
            this.elements = Object.fromEntries(
                Object.entries(selectors).map(([key, selector]) =>
                    [key, this.container.querySelector(selector)]
                )
            );
        }

        setupEventListeners() {
            const handlers = {
                releaseSlider: e => {
                    this.releaseTime = parseFloat(e.target.value);
                    this.elements.releaseValue.textContent = `${this.releaseTime}s`;
                },
                reverbSlider: e => {
                    const wetness = parseInt(e.target.value) / 100;
                    this.reverb.wet.value = wetness;
                    this.elements.reverbValue.textContent = `${e.target.value}%`;
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

        initializeEffects() {
            this.reverb = new Tone.Reverb({ decay: 1.5, wet: 0.5 }).toDestination();
        }

        async initializeSampler() {
            const availableNotes = ['C1', 'F#1', 'C2', 'F#2', 'C3', 'F#3', 'C4', 'F#4', 'C5', 'F#5'];
            const urls = Object.fromEntries(
                availableNotes
                    .map(note => [note, document.querySelector(`a[href*="${note}.wav"]`)?.href])
                    .filter(([, url]) => url)
            );

            if (!Object.keys(urls).length) {
                this.handleSamplerError();
                return;
            }

            this.sampler = new Tone.Sampler({
                urls,
                onload: () => this.handleSamplerLoad(),
            }).connect(this.reverb);
        }

        handleSamplerError() {
            console.log('No WAV files found');
            
        }

        handleSamplerLoad() {
            console.log('Sampler loaded');
            this.container.querySelectorAll('.key').forEach(key => key.style.opacity = '1');
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

    let container = document.getElementById('custom-player');
    if (!container) {
        container = document.createElement('div');
        container.id = 'custom-player';
        document.body.appendChild(container);
    }
    new KeyboardPlayer('custom-player');
}