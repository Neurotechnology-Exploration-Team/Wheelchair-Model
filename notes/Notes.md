# Notes
## Model

- RNN Model

## Data Loader

- Artifact removal
### Filtering

- bandpass filtering for bands alpha and beta 8hz-30hz or 1hz-60hz maybe use a GA to find optimal
- Spectral Analysis
- Feature Extraction
- Normalization

### Training Data
 
- Data will be given in large csv format of 
  - "Timestamp,EEG_1,EEG_2,EEG_3,EEG_4,EEG_5,EEG_6,EEG_7,EEG_8,Label"
- Label can also be in a separate file but has to be placed in training data frame
- Overlapping will be about 50% of the last frame so that it can be read 

### Live Data

- Data will be segmented in half seconds in groups of 125
- It'll be processed and de-noised then sent into model
- Predictions will be sent out every half second