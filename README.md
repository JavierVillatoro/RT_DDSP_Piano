# DDSP-Piano: Piano Synthesis via Neural Physical Modeling

This project involves the development of a high-fidelity polyphonic piano synthesizer based on the **DDSP (Differentiable Digital Signal Processing)** architecture. The main objective is to achieve a realistic synthesis that captures the timbral and physical subtleties of a grand piano, optimizing the model for real-time execution.

## 📖 Inspiration and References
This work builds upon and expands the original research by Google Magenta:
* **Original Paper:** *Renault, L., Mignot, R., & Roberts, A. (2022). (https://hal.science/hal-04073770v2/file/)*
* **Dataset:** [MAESTRO Dataset (v3.0.0)](https://magenta.tensorflow.org/datasets/maestro), specifically utilizing recordings and MIDI files from the year 2009 for training and validation.

## 🚀 Project Status and Results
The model has been trained following a hierarchical, phased learning scheme to ensure convergence and physical stability:

1.  **Phase 1 (Timbral Learning):** Successfully captured the spectral signature of the piano (hammer strike, harmonic content, and background noise). A multi-resolution spectral loss of **~6.36** was achieved.
2.  **Phase 2 (Physical Fine-Tuning):** Training of inharmonicity and micro-detuning parameters. An organic integration of acoustic beats (natural chorus) was achieved without compromising the fundamental tuning.

## 🛠 Engineering Modifications and Improvements
Building upon the paper's base architecture, the following improvements have been implemented to resolve critical issues detected during inference testing:

### 1. Scalable Polyphony (Dynamic Voice Allocation)
* **Problem:** Training was limited to 8 voices due to VRAM constraints. In complex pieces (e.g., Ravel's *Ondine*), the system suffered from "note stealing" and abrupt cutoffs.
* **Solution:** The polyphonic orchestrator was restructured using shared weights to allow inference of up to **24 simultaneous voices**. This eliminates cutoff artifacts and enables the rendering of high-density polyphonic pieces with complete fidelity.

### 2. Detuner Stabilization (Tuning Control)
* **Problem:** The original model exhibited instabilities leading to chaotic tunings ("honky-tonk" sound).
* **Solution:** A "physical safeguard" was applied via:
    * **Strict zero initialization:** Forcing weights in dense layers to zero to ensure the piano starts perfectly in tune.
    * **Excursion Restriction:** Mathematical limitation of the Detuner output to **±5 cents** ($\pm 0.05$ semitones). This allows for the warmth of acoustic beating while preventing audibly destructive detuning.

### 3. RNN Amnesia Diagnosis (Temporal Drift)
* **Finding:** During inference in Python with long MIDI files, timbral degradation was detected after 15-20 seconds. 
* **Diagnosis:** This was identified as a *Hidden State* management issue in *stateless* environments (TensorFlow/Python). By resetting the GRU's memory between blocks, the network loses its physical continuity. 
* **VST Solution:** This behavior was identified as a laboratory artifact that will disappear in the real-time phase thanks to the *stateful* nature of RTNeural.

## ⚠️ Tests Conducted
* **Test Renderings:** Complex works such as *Mompou* and passages by Ravel have been successfully rendered, validating timbral stability and correct voice allocation.
* **Success Metric:** Spectral consistency maintained and absence of cutoffs during the string's *note release* phase.

## ⏩ Current Phase: Live Implementation with RTNeural
After validating the sonic quality in the research environment (Python), the project enters its final integration phase:
1.  **Export:** Conversion of the trained weights (`.h5`) to a JSON format compatible with the **RTNeural** library.
2.  **C++ Architecture:** Implementation of the synthesis engine in a VST plugin using the **JUCE** framework.
3.  **State Management:** Leveraging the persistence of the hidden state in RTNeural to eliminate network amnesia and allow for infinite playback without degradation.

---
*This project is being developed as part of a Bachelor's Thesis (Trabajo de Fin de Grado - TFG) focused on the application of Deep Learning to Digital Signal Processing (DSP).*