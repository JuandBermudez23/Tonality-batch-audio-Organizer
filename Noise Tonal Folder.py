import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import soundfile as sf
import sounddevice as sd
import numpy as np
import threading
import librosa
from pathlib import Path

MIN_CHUNK_MS = 10
MAX_CHUNK_MS = 1000
CROSSFADE_MS = 10  # 10ms crossfade duration

class AudioTonalityOrganizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Tonality Reorganizer - Folder Mode")

        self.audio = None
        self.sorted_audio = None
        self.sr = None
        self.loaded_files = []

        self.order_var = tk.StringVar(value="Noisy → Tonal")

        # GUI
        tk.Button(root, text="Upload Folder", width=45, command=self.load_folder).pack(pady=4)
        
        # File list display
        self.file_listbox = tk.Listbox(root, height=6, width=60)
        self.file_listbox.pack(pady=4)
        
        tk.Button(root, text="Play Original Audio", width=45, command=self.play_original).pack(pady=4)

        order_frame = tk.Frame(root)
        order_frame.pack(pady=4)
        tk.Label(order_frame, text="Order:").pack(side="left", padx=5)
        tk.OptionMenu(order_frame, self.order_var,
                      "Noisy → Tonal", "Tonal → Noisy").pack(side="left")

        self.chunk_label = tk.Label(root, text="Chunk Size: 1000 ms")
        self.chunk_label.pack()
        self.chunk_slider = tk.Scale(root, from_=MIN_CHUNK_MS, to=MAX_CHUNK_MS,
                                     orient="horizontal", length=320, command=self.update_chunk_label)
        self.chunk_slider.set(1000)
        self.chunk_slider.pack(pady=4)

        tk.Button(root, text="Analyze & Reorganize", width=45, command=self.process_audio).pack(pady=6)
        tk.Button(root, text="Play Reorganized Audio", width=45, command=self.play_sorted).pack(pady=4)
        tk.Button(root, text="STOP Audio", width=45, fg="red", command=self.stop_audio).pack(pady=4)
        tk.Button(root, text="Export Reorganized Audio", width=45, command=self.export_audio).pack(pady=6)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=6)
        
        # Status label
        self.status_label = tk.Label(root, text="Ready", fg="blue")
        self.status_label.pack(pady=4)

    # --- GUI helpers ---
    def update_chunk_label(self, value):
        self.chunk_label.config(text=f"Chunk Size: {value} ms")

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Audio Files")
        if not folder_path:
            return

        # Supported audio formats
        audio_extensions = {'.wav', '.flac', '.aiff', '.aif', '.ogg', '.mp3'}
        
        # Find all audio files in the folder
        audio_files = []
        for file in Path(folder_path).iterdir():
            if file.suffix.lower() in audio_extensions and file.is_file():
                audio_files.append(file)
        
        if not audio_files:
            messagebox.showwarning("No Audio Files", "No supported audio files found in the selected folder.")
            return
        
        # Sort files by name for consistent ordering
        audio_files.sort()
        
        # Update file list display
        self.file_listbox.delete(0, tk.END)
        for file in audio_files:
            self.file_listbox.insert(tk.END, file.name)
        
        self.status_label.config(text=f"Loading {len(audio_files)} files...")
        self.root.update_idletasks()
        
        # Load and concatenate all audio files
        all_audio = []
        target_sr = None
        self.loaded_files = []
        
        for i, file_path in enumerate(audio_files):
            try:
                audio_data, sr = sf.read(str(file_path), always_2d=True, dtype="float32")
                
                # Set target sample rate from first file
                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    # Resample if needed
                    messagebox.showwarning(
                        "Sample Rate Mismatch",
                        f"{file_path.name} has different sample rate ({sr} vs {target_sr}). Skipping this file."
                    )
                    continue
                
                # Convert to stereo if mono
                if audio_data.shape[1] == 1:
                    audio_data = np.column_stack([audio_data, audio_data])
                
                all_audio.append(audio_data)
                self.loaded_files.append(file_path.name)
                
            except Exception as e:
                messagebox.showerror("Error Loading File", f"Could not load {file_path.name}:\n{str(e)}")
        
        if not all_audio:
            messagebox.showerror("Loading Failed", "No audio files could be loaded.")
            return
        
        # Concatenate all audio files
        self.audio = np.concatenate(all_audio, axis=0)
        self.sr = target_sr
        self.sorted_audio = None
        
        duration = len(self.audio) / self.sr
        messagebox.showinfo(
            "Loaded",
            f"Loaded {len(self.loaded_files)} files\n"
            f"Total Duration: {duration:.2f} seconds\n"
            f"Sample Rate: {self.sr} Hz"
        )
        
        self.status_label.config(text=f"{len(self.loaded_files)} files loaded ({duration:.2f}s)")

    # --- Multi-Feature Tonality Estimation ---
    def estimate_tonality(self, chunk):
        """
        Estimates tonality (harmonicity vs noisiness) using multiple features:
        - Spectral flatness: LOW flatness = tonal, HIGH flatness = noisy
        - Harmonic-to-noise ratio: measures harmonicity
        - Spectral contrast: tonal sounds have high contrast between peaks/valleys
        - Zero-crossing rate: LOW zcr = tonal, HIGH zcr = noisy
        
        Returns a value where:
        - Low values = noisy (white noise, breath, percussion)
        - High values = tonal (pitched instruments, singing)
        """
        mono = np.mean(chunk, axis=1).astype(np.float64)
        
        # Handle very short chunks
        if len(mono) < 2048:
            return 0.0
        
        try:
            # 1. Spectral flatness (LOW = tonal, HIGH = noisy)
            # We invert this so high values = tonal
            flatness = np.mean(librosa.feature.spectral_flatness(y=mono, n_fft=2048, hop_length=512))
            tonality_from_flatness = 1.0 - flatness  # Invert: low flatness = high tonality
            
            # 2. Harmonic-to-noise ratio using harmonic/percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(mono)
            harmonic_energy = np.sum(y_harmonic**2)
            percussive_energy = np.sum(y_percussive**2)
            total_energy = harmonic_energy + percussive_energy + 1e-10
            harmonic_ratio = harmonic_energy / total_energy
            
            # 3. Spectral contrast (tonal sounds have higher contrast)
            contrast = librosa.feature.spectral_contrast(y=mono, sr=self.sr, n_fft=2048, hop_length=512)
            # Normalize contrast (typically ranges from -40 to +40 dB)
            contrast_mean = np.mean(contrast)
            contrast_norm = (contrast_mean + 40) / 80  # Normalize to 0-1 range
            contrast_norm = np.clip(contrast_norm, 0, 1)
            
            # 4. Zero-crossing rate (LOW = tonal, HIGH = noisy)
            # We invert this so high values = tonal
            zcr = np.mean(librosa.feature.zero_crossing_rate(mono, frame_length=2048, hop_length=512))
            # ZCR typically ranges from 0 to 0.5, normalize and invert
            tonality_from_zcr = 1.0 - (zcr * 2.0)  # Invert and normalize
            tonality_from_zcr = np.clip(tonality_from_zcr, 0, 1)
            
            # 5. Spectral peak prominence (tonal sounds have prominent peaks)
            S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
            S_mean = np.mean(S, axis=1)
            S_std = np.std(S, axis=1)
            peak_prominence = np.mean(S_std / (S_mean + 1e-10))
            peak_prominence_norm = np.clip(peak_prominence / 2.0, 0, 1)
            
            # Weighted combination
            # Harmonic ratio and flatness are most important
            tonality = (harmonic_ratio * 0.35 +           # Harmonic content (most important)
                       tonality_from_flatness * 0.25 +    # Spectral flatness (inverted)
                       contrast_norm * 0.20 +             # Spectral contrast
                       tonality_from_zcr * 0.15 +         # Zero-crossing (inverted)
                       peak_prominence_norm * 0.05)       # Peak prominence
            
            return float(tonality)
            
        except Exception as e:
            print(f"Error calculating tonality: {e}")
            return 0.0

    # --- Crossfade implementation ---
    def apply_crossfade(self, chunks, crossfade_samples):
        """Apply crossfade between consecutive chunks with proper overlap"""
        if len(chunks) == 0:
            return np.array([]).reshape(0, 2)
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Pre-calculate total length needed
        total_length = sum(len(chunk) for chunk in chunks)
        total_length -= (len(chunks) - 1) * crossfade_samples  # Account for overlaps
        
        num_channels = chunks[0].shape[1]
        result = np.zeros((total_length, num_channels), dtype=np.float32)
        
        # Create fade curves once
        fade_out = np.linspace(1, 0, crossfade_samples).reshape(-1, 1)
        fade_in = np.linspace(0, 1, crossfade_samples).reshape(-1, 1)
        
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            
            # Adjust crossfade for chunks shorter than crossfade length
            actual_crossfade = min(crossfade_samples, chunk_len // 2)
            
            if i == 0:
                # First chunk: copy entirely
                result[current_pos:current_pos + chunk_len] = chunk
                current_pos += chunk_len
            else:
                # Adjust fade curves if needed
                if actual_crossfade < crossfade_samples:
                    fade_out_local = np.linspace(1, 0, actual_crossfade).reshape(-1, 1)
                    fade_in_local = np.linspace(0, 1, actual_crossfade).reshape(-1, 1)
                else:
                    fade_out_local = fade_out
                    fade_in_local = fade_in
                
                # Crossfade with previous chunk's tail
                crossfade_start = current_pos - actual_crossfade
                result[crossfade_start:current_pos] *= fade_out_local
                result[crossfade_start:current_pos] += chunk[:actual_crossfade] * fade_in_local
                
                # Add remainder of current chunk
                result[current_pos:current_pos + chunk_len - actual_crossfade] = chunk[actual_crossfade:]
                current_pos += chunk_len - actual_crossfade
        
        return result

    # --- Process and reorder audio ---
    def process_audio(self):
        if self.audio is None:
            messagebox.showwarning("No Audio", "Upload a folder with audio files first.")
            return

        self.status_label.config(text="Processing...")
        self.root.update_idletasks()

        chunk_ms = self.chunk_slider.get()
        chunk_size = int(self.sr * chunk_ms / 1000.0)
        total_chunks = int(np.ceil(len(self.audio) / chunk_size))
        self.progress["maximum"] = total_chunks
        self.progress["value"] = 0

        chunks = []
        processed = 0
        tonality_values = []  # DEBUG

        print(f"\nAnalyzing {total_chunks} chunks of {chunk_ms}ms each...")

        for start in range(0, len(self.audio), chunk_size):
            chunk = self.audio[start:start+chunk_size]
            if len(chunk) == 0:
                continue
            
            tonality = self.estimate_tonality(chunk)
            tonality_values.append(tonality)
            chunks.append((tonality, chunk))
            processed += 1
            self.progress["value"] = processed
            self.root.update_idletasks()

        # DEBUG: Print tonality statistics
        print("="*60)
        print("TONALITY ANALYSIS:")
        print(f"Total chunks analyzed: {len(tonality_values)}")
        print(f"Min tonality: {min(tonality_values):.6f} (most noisy)")
        print(f"Max tonality: {max(tonality_values):.6f} (most tonal)")
        print(f"Mean tonality: {np.mean(tonality_values):.6f}")
        print(f"Std deviation: {np.std(tonality_values):.6f}")
        print(f"Range: {max(tonality_values) - min(tonality_values):.6f}")
        print(f"\nFirst 10 values: {[f'{v:.4f}' for v in tonality_values[:10]]}")
        print(f"Last 10 values: {[f'{v:.4f}' for v in tonality_values[-10:]]}")

        # Check if there's actual variation
        if np.std(tonality_values) < 0.001:
            print("\n⚠️  WARNING: Very low variation in tonality values!")
            print("   The audio might be too uniform or the chunk size too small.")
        
        reverse = self.order_var.get() == "Tonal → Noisy"
        chunks.sort(key=lambda x: x[0], reverse=reverse)

        # DEBUG: Print sorted order
        sorted_tonality = [t for t, _ in chunks]
        print(f"\nAFTER SORTING ({self.order_var.get()}):")
        print(f"First 10 tonality values: {[f'{v:.4f}' for v in sorted_tonality[:10]]}")
        print(f"Last 10 tonality values: {[f'{v:.4f}' for v in sorted_tonality[-10:]]}")
        print("="*60 + "\n")

        # Extract just the audio chunks and apply crossfading
        sorted_chunks = [chunk for _, chunk in chunks]
        crossfade_samples = int(self.sr * (CROSSFADE_MS / 1000.0))
        
        # Apply crossfading with overlap
        self.sorted_audio = self.apply_crossfade(sorted_chunks, crossfade_samples)
        
        duration = len(self.sorted_audio) / self.sr
        variation_msg = ""
        if np.std(tonality_values) < 0.001:
            variation_msg = "\n\nNote: Low tonality variation detected.\nTry adjusting chunk size or check audio content."
        
        messagebox.showinfo("Done", 
            f"Reorganized {len(self.loaded_files)} files using {chunk_ms} ms chunks\n"
            f"Order: {self.order_var.get()}\n"
            f"Crossfade: {CROSSFADE_MS} ms\n"
            f"Tonality range: {min(tonality_values):.4f} to {max(tonality_values):.4f}\n"
            f"Output Duration: {duration:.2f} seconds"
            f"{variation_msg}")
        
        self.status_label.config(text=f"Reorganized: {duration:.2f}s")

    # --- Playback ---
    def play_audio(self, audio):
        if audio is None or len(audio) == 0:
            return
        sd.stop()
        
        # Ensure proper shape
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        
        # Play with explicit parameters
        sd.play(audio.astype(np.float32), samplerate=self.sr)
        sd.wait()

    def play_original(self):
        if self.audio is not None:
            threading.Thread(target=self.play_audio, args=(self.audio,), daemon=True).start()
        else:
            messagebox.showwarning("No Audio", "Upload a folder first.")

    def play_sorted(self):
        if self.sorted_audio is not None and len(self.sorted_audio) > 0:
            threading.Thread(target=self.play_audio, args=(self.sorted_audio,), daemon=True).start()
        else:
            messagebox.showwarning("No Audio", "Analyze audio first.")

    # --- Stop playback ---
    def stop_audio(self):
        sd.stop()

    # --- Export ---
    def export_audio(self):
        if self.sorted_audio is None:
            messagebox.showwarning("Nothing to Export", "Analyze audio first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav",
                                                 filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac"), ("AIFF", "*.aiff")])
        if save_path:
            sf.write(save_path, self.sorted_audio, self.sr)
            messagebox.showinfo("Exported", f"Saved to:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioTonalityOrganizer(root)
    root.mainloop()