\# NeuroLat: EEG Seizure Detection and Lateralization



\## Overview



NeuroLat is a research project for electroencephalography (EEG) analysis using deep learning. It combines a custom EEGConformer architecture with spatial attention to support seizure detection and seizure lateralization.



The project includes trained model weights, normalization parameters, a training script, and a web interface.



\*\*Research use only:\*\* NeuroLat is not a clinically validated diagnostic system.



\## Objectives



\* \*\*M1 — Seizure Detection:\*\* identify seizure-related EEG segments.

\* \*\*M2 — Seizure Lateralization:\*\* estimate seizure lateralization according to the classes supported by the model.



\## Dataset



The project uses the Temple University Hospital EEG Seizure Corpus (TUSZ), version 2.0.3.



| Parameter          | Configuration     |

| ------------------ | ----------------- |

| Sampling frequency | 250 Hz            |

| Window duration    | 4 seconds         |

| Samples per window | 1,000 per channel |

| EEG channels       | 19                |



The dataset is not included in this repository. Users must obtain it separately and comply with the dataset provider's terms.



\## Preprocessing



The implemented pipeline includes:



\* Band-pass filtering from 0.5 to 70 Hz.

\* A 60 Hz notch filter.

\* Reconstruction of 19 EEG channels.

\* Segmentation into 4-second windows.

\* Normalization using saved parameters.



Refer to the source code for the exact implementation and channel configuration.



\## Model Architecture



NeuroLat uses a custom EEGConformer architecture with spatial attention. Its documented components include:



\* Spatial attention for channel-wise weighting.

\* Temporal patch embedding.

\* Spatial convolution.

\* Four Conformer blocks.

\* A classification head.



The documented configuration uses a 64-dimensional representation and four attention heads. The architecture is custom-built; no pretrained initialization is claimed here.



\## Repository Structure



```text

NeuroLat/

└── PFE\_PARTAGE\_finale/

&#x20;   ├── app.py

&#x20;   ├── models/

&#x20;   │   ├── model\_m1\_v2.pt

&#x20;   │   ├── model\_m2\_v3.pt

&#x20;   │   ├── mu1\_v2.npy

&#x20;   │   ├── mu2\_v3.npy

&#x20;   │   ├── std1\_v2.npy

&#x20;   │   └── std2\_v3.npy

&#x20;   ├── training/

&#x20;   │   └── entrainement\_M1\_M2\_EEGConformer.py

&#x20;   └── neuro-lat-eeg-interface/

```



\## Model Files



| File                        | Purpose                     |

| --------------------------- | --------------------------- |

| `model\_m1\_v2.pt`            | Trained M1 model            |

| `model\_m2\_v3.pt`            | Trained M2 model            |

| `mu1\_v2.npy`, `std1\_v2.npy` | M1 normalization parameters |

| `mu2\_v3.npy`, `std2\_v3.npy` | M2 normalization parameters |



Use these files with the matching model implementation and preprocessing pipeline.



\## Installation



Create and activate a Python virtual environment:



```bash

python -m venv .venv

```



Install the dependencies listed in `requirements.txt` once that file has been prepared and tested:



```bash

pip install -r requirements.txt

```



Run the Python application from the directory containing `app.py`:



```bash

python app.py

```



For the Next.js interface, open a terminal in the frontend directory:



```bash

npm install

npm run dev

```



Consult the source code and `package.json` for the exact runtime requirements.



\## Training and Reproducibility



The training script is located at:



`PFE\_PARTAGE\_finale/training/entrainement\_M1\_M2\_EEGConformer.py`



Reproducing the experiments requires the appropriate dataset, compatible dependencies, preprocessing settings, model configuration, and evaluation procedure.



Random seeds, dataset partitions, classification thresholds, hyperparameters, and software versions should be documented where available.



\## Experimental Results



Verified quantitative results have not yet been added to this README. Performance claims should be reported together with the evaluation protocol, sample counts, and relevant metrics.



\## Limitations



\* Performance depends on the dataset and evaluation protocol.

\* Generalization to other EEG datasets requires further validation.

\* Lateralization predictions depend on the model's label definitions.

\* The availability of trained weights does not guarantee full reproducibility.

\* NeuroLat is a research prototype, not a validated medical device.



\## Ethical Considerations



Do not publish raw patient recordings or confidential information without appropriate authorization. Verify the rights to distribute the code, trained weights, and derived artifacts before publication.



\## License



A license has not yet been specified. Add an appropriate license only after confirming the rights to distribute the code and model weights.



\## Citation



Citation information will be added when the associated scientific publication and bibliographic details are available.



\## Disclaimer



NeuroLat is provided for research and educational purposes only. It should not be used as the sole basis for medical diagnosis or treatment decisions.



