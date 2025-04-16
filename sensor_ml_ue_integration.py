# ignore this file, use mahata_integration.py instead - it is the updated version

# Imports
import csv
import pandas as pd
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import pickle
import joblib
from collections import deque
import numpy as np
from scipy import signal
from scipy import stats
from scipy.ndimage import label as sci_label
import warnings
import queue
import warnings
import threading
import time

###################################################################################################
# Global Configuration and Buffer Setup
###################################################################################################
sampling_rate = 100  # in Hz
buffer_size = sampling_rate * 5  # 5-second buffer (500 samples)
ecg_buffer = deque(maxlen=buffer_size)
timestamps = []

# Global variables to store the latest stress and valence predictions
latest_stress = 'Invalid'
latest_valence = 'Invalid'

# Paths for models and output files
ecg_stress_model_path = r"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\ECG_STRESS\random_forest_model.pkl"
ecg_valence_model_path = r"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\ECG_VALENCE\best_valence_model.joblib"

# Output CSV for predictions (header now includes textual labels)
timestamp_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_csv = rf"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\predictions_{timestamp_now}.csv"
# Create CSV with headers
df = pd.DataFrame(columns=["START_TIME", "END_TIME", "DURATION", "STRESS LEVEL", "VALENCE LEVEL"])
df.to_csv(output_csv, index=False)

###################################################################################################
# Peak Detection and Feature Extraction Functions
###################################################################################################

def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
    '''
    Peak detection algorithm using cross correlation and threshold 
    '''
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)
    
    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return np.where(similarity > threshold)[0], similarity

def group_peaks(p, threshold=5):
    '''
    The peak detection algorithm finds multiple peaks for each QRS complex. 
    Here we group collections of peaks that are very near (within threshold) and we take the median index 
    '''
    # initialize output
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = sci_label(np.diff(p) < threshold)
 
    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output

def calc_sdsd(ecg_signal):
    # detect peaks
    peaks, similarity = detect_peaks(ecg_signal, threshold=0.3)
    # group peaks so we get a single peak per beat
    grouped_peaks = group_peaks(peaks)
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    # successive differences
    diff_rr = np.diff(rr)
    return np.std(diff_rr)

def calc_sdrr_rmssd(ecg_signal):
    # detect peaks
    peaks, similarity = detect_peaks(ecg_signal, threshold=0.3)
    # group peaks so we get a single peak per beat
    grouped_peaks = group_peaks(peaks)
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    # Calculate SDRR
    sdrr = np.std(rr)
    # Calculate RMSSD
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    if rmssd == 0:
        return 0
    return sdrr / rmssd

def calc_pnnx(ecg_signal, threshold=25):  # pNN25
    # detect peaks
    peaks, similarity = detect_peaks(ecg_signal, threshold=0.3)
    # group peaks so we get a single peak per beat
    grouped_peaks = group_peaks(peaks)
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    # Count number of successive intervals that differ by more than threshold
    nn_count = np.sum(np.abs(np.diff(rr)) > threshold)
    # Calculate percentage
    if len(rr) <= 1:
        return 0
    return 100 * nn_count / (len(rr) - 1)

def calc_sd1(ecg_signal):
    # detect peaks
    peaks, similarity = detect_peaks(ecg_signal, threshold=0.3)
    # group peaks so we get a single peak per beat
    grouped_peaks = group_peaks(peaks)
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    # Calculate SD1
    diff_rr = np.diff(rr)
    return np.sqrt(np.std(diff_rr) ** 2 * 0.5)

def calc_skew(ecg_signal):
    # detect peaks
    peaks, similarity = detect_peaks(ecg_signal, threshold=0.3)
    # group peaks so we get a single peak per beat
    grouped_peaks = group_peaks(peaks)
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    return stats.skew(rr)

def calc_relative_rr(ecg_signal):
    # detect peaks
    peaks, similarity = detect_peaks(ecg_signal, threshold=0.3)
    # group peaks so we get a single peak per beat
    grouped_peaks = group_peaks(peaks)
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    # Normalize RR intervals to the mean
    if len(rr) == 0 or np.mean(rr) == 0:
        return np.array([])
    rel_rr = rr / np.mean(rr)
    return rel_rr

def calc_sdrr_rmssd_rel_rr(ecg_signal):
    rel_rr = calc_relative_rr(ecg_signal)
    if len(rel_rr) <= 1:
        return 0
    # Calculate SDRR of relative RR
    sdrr_rel = np.std(rel_rr)
    # Calculate RMSSD of relative RR
    diff_rel_rr = np.diff(rel_rr)
    rmssd_rel = np.sqrt(np.mean(diff_rel_rr ** 2))
    if rmssd_rel == 0:
        return 0
    return sdrr_rel / rmssd_rel

def calc_skew_rel_rr(ecg_signal):
    rel_rr = calc_relative_rr(ecg_signal)
    if len(rel_rr) <= 1:
        return 0
    return stats.skew(rel_rr)

###################################################################################################
# Classes for ECG Processing with Quality Checks
###################################################################################################

class ECGStressProcessor:
    def __init__(self, model):
        self.model = model
        # Scale factor to convert raw ECG data to training range (±0.04 to ~800)
        self.ecg_scale_factor = 20000

    def is_valid_data(self, data):
        if len(data) == 0:
            return False
        return not (np.any(np.isnan(data)) or np.any(np.isinf(data)))
    
    def scale_ecg_signal(self, ecg_signal):
        return ecg_signal * self.ecg_scale_factor

    def check_quality(self, ecg_signal):
        """
        ECG Quality Checks:
          - Standard deviation must be at least 0.01.
          - The power spectrum should not be dominated (>80%) by one frequency.
        """
        if np.std(ecg_signal) < 0.01:
            print("ECG quality check failed: low variance")
            return False
        fft = np.fft.rfft(ecg_signal)
        power = np.abs(fft) ** 2
        peak_power = np.max(power)
        total_power = np.sum(power)
        if total_power > 0 and (peak_power / total_power) > 0.8:
            print("ECG quality check failed: dominant single frequency")
            return False
        return True

    def calculate_rr_intervals(self, scaled_signal, sampling_rate=100):
        try:
            r_peaks, _ = signal.find_peaks(scaled_signal, distance=sampling_rate//2)
            if len(r_peaks) < 2:
                return np.array([])
            rr_intervals = np.diff(r_peaks) * (1000 / sampling_rate)
            return rr_intervals
        except Exception as e:
            print(f"Error in calculate_rr_intervals: {e}")
            return np.array([])

    def extract_features(self, ecg_signal, sampling_rate=100):
        if not self.is_valid_data(ecg_signal):
            return None
        if not self.check_quality(ecg_signal):
            return None
        try:
            scaled_signal = self.scale_ecg_signal(ecg_signal)
            rr_intervals = self.calculate_rr_intervals(scaled_signal, sampling_rate)
            if len(rr_intervals) < 2:
                return None
            rel_rr = np.diff(rr_intervals)
            features = {
                'MEAN_RR': np.mean(rr_intervals),
                'MEDIAN_RR': np.median(rr_intervals),
                'SDRR_RMSSD': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
                'MEDIAN_REL_RR': np.median(rel_rr) if len(rel_rr) > 0 else 0,
                'SDRR_RMSSD_REL_RR': np.sqrt(np.mean(np.square(np.diff(rel_rr)))) if len(rel_rr) > 1 else 0,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freqs, psd = signal.welch(rr_intervals, fs=sampling_rate, nperseg=len(rr_intervals))
                vlf_mask = freqs <= 0.04
                total_power = np.sum(psd)
                if total_power > 0:
                    features['VLF'] = np.sum(psd[vlf_mask])
                    features['VLF_PCT'] = features['VLF'] / total_power
                else:
                    features['VLF'] = 0
                    features['VLF_PCT'] = 0
            if any(np.isnan(list(features.values()))) or any(np.isinf(list(features.values()))):
                return None
            return features
        except Exception as e:
            print(f"Error in extract_features (ECG): {e}")
            return None
        
    def preprocess_and_predict(self, ecg_signal):
        try:
            features = self.extract_features(ecg_signal, sampling_rate)
            if features is None:
                return 'Invalid', None
            feature_df = np.array([list(features.values())]).reshape(1, -1)
            if np.any(np.isnan(feature_df)) or np.any(np.isinf(feature_df)):
                return 'Invalid', None
            prediction = self.model.predict(feature_df)[0]
            return prediction, features
        except Exception as e:
            print(f"Error in preprocess_and_predict (ECG): {e}")
            return 'Invalid', None
class ECGValenceProcessor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # Scale factor to convert raw ECG data to training range
        # Real-time data range: ~±0.34, Training data range: ~±20
        self.ecg_scale_factor = 60  # Approximated scale factor
        print(f"Valence model loaded successfully from {model_path}")

    def is_valid_data(self, data):
        if len(data) == 0:
            return False
        return not (np.any(np.isnan(data)) or np.any(np.isinf(data)))

    def scale_ecg_signal(self, ecg_signal):
        """
        Scale the ECG signal to match the training data range
        """
        return ecg_signal * self.ecg_scale_factor

    def check_quality(self, ecg_signal):
        """
        ECG Quality Checks:
          - Standard deviation must be at least 0.01.
          - At least 2 peaks must be detected for RR interval calculation.
        """
        if np.std(ecg_signal) < 0.01:
            print("ECG quality check failed: low variance")
            return False
            
        # Check if we can detect at least two peaks
        peaks, _ = detect_peaks(ecg_signal, threshold=0.3)
        grouped_peaks = group_peaks(peaks)
        if len(grouped_peaks) < 2:
            print("ECG quality check failed: insufficient peaks detected")
            return False
            
        return True

    def extract_features(self, ecg_signal):
        """
        Extract all required features for valence prediction
        """
        if not self.is_valid_data(ecg_signal):
            return None
            
        if not self.check_quality(ecg_signal):
            return None
            
        try:
            # Scale the ECG signal before feature extraction
            scaled_signal = self.scale_ecg_signal(ecg_signal)
            
            features = {
                "SDSD": calc_sdsd(scaled_signal),
                "SDRR_RMSSD": calc_sdrr_rmssd(scaled_signal),
                "pNN25": calc_pnnx(scaled_signal, threshold=25),
                "SD1": calc_sd1(scaled_signal),
                "SKEW": calc_skew(scaled_signal),
                "SDRR_RMSSD_REL_RR": calc_sdrr_rmssd_rel_rr(scaled_signal),
                "SKEW_REL_RR": calc_skew_rel_rr(scaled_signal)
            }
            
            if any(np.isnan(list(features.values()))) or any(np.isinf(list(features.values()))):
                return None
                
            return features
            
        except Exception as e:
            print(f"Error in extract_features (ECG Valence): {e}")
            return None

    def preprocess_and_predict(self, ecg_signal):
        try:
            features = self.extract_features(ecg_signal)
            if features is None:
                return 'Invalid', None, None
                
            features_df = pd.DataFrame([features])
            
            if features_df.isnull().any().any() or np.any(np.isinf(features_df.values)):
                return 'Invalid', None, None
                
            prediction = self.model.predict(features_df)[0]
            prediction_proba = None
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features_df)[0]
                
            return prediction, prediction_proba, features
            
        except Exception as e:
            print(f"Error in preprocess_and_predict (ECG Valence): {e}")
            return 'Invalid', None, None

# Initialize processors
ecg_stress_processor = ECGStressProcessor(joblib.load(ecg_stress_model_path))
ecg_valence_processor = ECGValenceProcessor(ecg_valence_model_path)

###################################################################################################
# Utility Functions (Data Acquisition, Processing, Logging)
###################################################################################################

def new_session():
    """
    Creates new CSV file for raw data logging.
    """
    global csv_file_path_ECG
    csv_file_path_ECG = generate_filename("bitalino_data_ECG", "csv")
    with open(csv_file_path_ECG, "w", newline='') as csvfile1:
        csv_writer = csv.writer(csvfile1)
        csv_writer.writerow(["timestamp", "ECG"])

def generate_filename(base_name, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def acquire_sample(inlet, ecg_buffer, timestamps):
    """
    Pulls a sample from the inlet and appends data to the ECG buffer.
    Uses the system's current time as the timestamp.
    """
    samples, _ = inlet.pull_sample()  # Ignore LSL timestamp
    ts = datetime.now().timestamp()    # Use current system time
    ecg_buffer.append(samples[1])      # Assuming ECG data is at index 1
    timestamps.append(ts)
    return samples, ts

def process_buffers(ecg_buffer, timestamps):
    """
    Converts buffer to NumPy array and runs the predictions for both stress and valence.
    """
    ecg_signal = np.array(list(ecg_buffer))
    
    # Process for stress prediction
    stress_prediction, stress_features = ecg_stress_processor.preprocess_and_predict(ecg_signal)
    
    # Process for valence prediction
    valence_prediction, valence_probabilities, valence_features = ecg_valence_processor.preprocess_and_predict(ecg_signal)
    print(valence_prediction) # gets printed like 'High Valence' modify the mapping accordingly
    
    return stress_prediction, valence_prediction, stress_features, valence_features, list(timestamps)

def append_to_csv(file_path, new_row):
    """
    Appends a row to a CSV file.
    """
    with open(file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(new_row)

def log_predictions(output_csv, start_datetime, end_datetime, stress_prediction, valence_prediction):
    """
    Logs the predictions into a CSV file.
    Also calculates the duration of each processing bucket.
    Note: Textual labels are logged.
    """
    # Textual mapping for display purposes
    stress_map = {0: 'low', 1: 'medium', 2: 'high', 'Invalid': 'Invalid'}
    # Valence mapping
    # valence_map = {0: 'high', 1: 'medium', 2: 'low', 'Invalid': 'Invalid'}


        # Updated valence mapping to handle string outputs like 'High Valence'
    if isinstance(valence_prediction, str):
        # Parse the textual prediction (e.g., 'High Valence' -> 'high')
        if 'High' in valence_prediction:
            mapped_valence = 'high'
        elif 'Medium' in valence_prediction:
            mapped_valence = 'medium'
        elif 'Low' in valence_prediction:
            mapped_valence = 'low'
        else:
            mapped_valence = 'Invalid'

    
    mapped_stress = stress_map.get(stress_prediction, 'Invalid')
    #mapped_valence = valence_map.get(valence_prediction, 'Invalid')
    
    # Calculate the duration of this bucket in seconds
    duration = (end_datetime - start_datetime).total_seconds()
    
    new_entry = pd.DataFrame([{
        "START_TIME": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "END_TIME": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "DURATION": duration,
        "STRESS LEVEL": mapped_stress,
        "VALENCE LEVEL": mapped_valence,
    }])
    new_entry.to_csv(output_csv, mode='a', header=False, index=False)
    print("inside log predictions ")
    print(mapped_stress, mapped_valence)
    print("inside log predictions ^")

    return mapped_stress, mapped_valence

def write_raw_data(timestamp, samples):
    """
    Logs raw ECG samples to CSV file by appending new lines.
    """
    append_to_csv(csv_file_path_ECG, [timestamp, samples[1]])  # Assuming ECG is at index 1

def get_stress_valence():
    """
    Logs the predictions into a CSV file.
    Also calculates the duration of each processing bucket.
    Note: Textual labels are logged.
    """
    samples, ts = acquire_sample(inlet, ecg_buffer, timestamps)
    stress_prediction, valence_prediction, _, _, ts_buffer = process_buffers(ecg_buffer, timestamps)
    # Textual mapping for display purposes
    stress_map = {0: 'low', 1: 'medium', 2: 'high', 'Invalid': 'Invalid'}
    # Valence mapping
    #valence_map = {0: 'high', 1: 'medium', 2: 'low', 'Invalid': 'Invalid'}


        # Updated valence mapping to handle string outputs like 'High Valence'
    if isinstance(valence_prediction, str):
        # Parse the textual prediction (e.g., 'High Valence' -> 'high')
        if 'High' in valence_prediction:
            mapped_valence = 'high'
        elif 'Medium' in valence_prediction:
            mapped_valence = 'medium'
        elif 'Low' in valence_prediction:
            mapped_valence = 'low'
        else:
            mapped_valence = 'Invalid'

    
    mapped_stress = stress_map.get(stress_prediction, 'Invalid')
    #mapped_valence = valence_map.get(valence_prediction, 'Invalid')
    

    
    new_entry = pd.DataFrame([{
        "STRESS LEVEL": mapped_stress,
        "VALENCE LEVEL": mapped_valence,
    }])
    #new_entry.to_csv(output_csv, mode='a', header=False, index=False)
    print("inside gett_stress_valence ***")
    print(ecg_buffer)
    print(f"Stress Prediction: {stress_prediction} || Valence Prediction: {valence_prediction}")
    #print(stress_prediction,valence_prediction)
    print(f"Mapped Stress: {mapped_stress} || Mapped Valence: {mapped_valence}")
    #print(mapped_stress, mapped_valence)
    print("inside gett_stress_valence  ^^^")
    return mapped_stress, mapped_valence

###################################################################################################
# Data Preprocessor
###################################################################################################
class DataProcessor:
    def __init__(self, mac_address, sampling_rate=100):
        # Stream and buffer configuration
        self.sampling_rate = sampling_rate
        self.stress_buffer_size = sampling_rate * 5  # 5-second buffer
        self.valence_buffer_size = sampling_rate * 30  # 30-second buffer

        # Thread-safe queues for sharing data
        self.prediction_queue = queue.Queue()
        
        # Buffers and timestamps
        self.ecg_stress_buffer = deque(maxlen=self.stress_buffer_size)
        self.ecg_valence_buffer = deque(maxlen=self.valence_buffer_size)
        self.stress_timestamps = []
        self.valence_timestamps = []

        # Paths and output files
        self.ecg_stress_model_path = r"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\ECG_STRESS\random_forest_model.pkl"
        self.ecg_valence_model_path = r"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\ECG_VALENCE\best_valence_model.joblib"
        
        timestamp_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stress_output_csv = rf"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\predictions_stress_{timestamp_now}.csv"
        self.valence_output_csv = rf"C:\Users\BPA-POC\Desktop\Sree Devi ML Models\predictions_valence_{timestamp_now}.csv"

        # Initialize processors
        self.stress_processor = ECGStressProcessor(joblib.load(self.ecg_stress_model_path))
        self.valence_processor = ECGValenceProcessor(self.ecg_valence_model_path)

        # Initialize stream
        os_stream = resolve_stream("type", mac_address)
        if not os_stream:
            raise RuntimeError(f"No stream found for device with MAC address {mac_address}.")
        self.inlet = StreamInlet(os_stream[0])

        # Initialize CSV for raw data
        self.csv_file_path_ECG = self.generate_filename("bitalino_data_ECG", "csv")
        with open(self.csv_file_path_ECG, "w", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["timestamp", "ECG"])

    def generate_filename(self, base_name, extension):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"

    def acquire_sample(self):
        """
        Pulls a sample from the inlet and appends data to both stress and valence buffers.
        """
        samples, _ = self.inlet.pull_sample()  # Ignore LSL timestamp
        ts = datetime.now().timestamp()    # Use current system time
        
        # Append to stress buffer
        self.ecg_stress_buffer.append(samples[1])
        self.stress_timestamps.append(ts)
        
        # Append to valence buffer
        self.ecg_valence_buffer.append(samples[1])
        self.valence_timestamps.append(ts)
        
        return samples, ts

    def write_raw_data(self, timestamp, samples):
        """
        Logs raw ECG samples to CSV file by appending new lines.
        """
        with open(self.csv_file_path_ECG, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([timestamp, samples[1]])

    def process_buffers(self):
        """
        Processes both stress and valence buffers separately.
        """
        # Process stress buffer (5 seconds)
        stress_signal = np.array(list(self.ecg_stress_buffer))
        stress_prediction, stress_features = self.stress_processor.preprocess_and_predict(stress_signal)
        
        # Process valence buffer only when it's full (30 seconds)
        valence_prediction = 'Invalid'
        valence_features = None
        if len(self.ecg_valence_buffer) == self.valence_buffer_size:
            valence_signal = np.array(list(self.ecg_valence_buffer))
            valence_prediction, valence_probabilities, valence_features = self.valence_processor.preprocess_and_predict(valence_signal)
        
        return (stress_prediction, valence_prediction, 
                stress_features, valence_features, 
                list(self.stress_timestamps), list(self.valence_timestamps))

    def log_predictions(self, 
                        start_datetime_stress, end_datetime_stress, 
                        start_datetime_valence, end_datetime_valence,
                        stress_prediction, valence_prediction):
        """
        Logs predictions to separate CSV files.
        """
        # Stress mapping
        stress_map = {0: 'low', 1: 'medium', 2: 'high', 'Invalid': 'Invalid'}
        mapped_stress = stress_map.get(stress_prediction, 'Invalid')
        
        # Valence mapping
        if isinstance(valence_prediction, str):
            if 'High' in valence_prediction:
                mapped_valence = 'high'
            elif 'Medium' in valence_prediction:
                mapped_valence = 'medium'
            elif 'Low' in valence_prediction:
                mapped_valence = 'low'
            else:
                mapped_valence = 'Invalid'
        
        # Put the predictions in the queue for other threads to consume
        self.prediction_queue.put({
            'stress': mapped_stress,
            'valence': mapped_valence
        })
        
        # Stress predictions (5 seconds)
        stress_duration = (end_datetime_stress - start_datetime_stress).total_seconds()
        stress_entry = pd.DataFrame([{
            "START_TIME": start_datetime_stress.strftime("%Y-%m-%d %H:%M:%S"),
            "END_TIME": end_datetime_stress.strftime("%Y-%m-%d %H:%M:%S"),
            "DURATION": stress_duration,
            "STRESS LEVEL": mapped_stress,
        }])
        stress_entry.to_csv(self.stress_output_csv, mode='a', header=False, index=False)
        
        # Valence predictions (30 seconds, but only if a valid prediction is made)
        if mapped_valence != 'Invalid':
            valence_duration = (end_datetime_valence - start_datetime_valence).total_seconds()
            valence_entry = pd.DataFrame([{
                "START_TIME": start_datetime_valence.strftime("%Y-%m-%d %H:%M:%S"),
                "END_TIME": end_datetime_valence.strftime("%Y-%m-%d %H:%M:%S"),
                "DURATION": valence_duration,
                "VALENCE LEVEL": mapped_valence,
            }])
            valence_entry.to_csv(self.valence_output_csv, mode='a', header=False, index=False)

        print("inside gett_stress_valence ***")
        #print(ecg_buffer)
        print(f"Stress Prediction: {stress_prediction} || Valence Prediction: {valence_prediction}")
        #print(stress_prediction,valence_prediction)
        print(f"Mapped Stress: {mapped_stress} || Mapped Valence: {mapped_valence}")
        #print(mapped_stress, mapped_valence)
        print("inside gett_stress_valence  ^^^")

    def main_loop(self):
        """
        Main loop to acquire data, process buffers, and log predictions.
        """
        while True:
            try:
                # Acquire a sample
                samples, ts = self.acquire_sample()
                self.write_raw_data(ts, samples)
                
                # Process stress buffer when full (5 seconds)
                if len(self.ecg_stress_buffer) == self.stress_buffer_size:
                    stress_pred, valence_pred, _, _, stress_ts_buffer, valence_ts_buffer = self.process_buffers()
                    
                    stress_start_datetime = datetime.fromtimestamp(stress_ts_buffer[0])
                    stress_end_datetime = datetime.fromtimestamp(stress_ts_buffer[-1])
                    
                    # If valence buffer is full, get its timestamps
                    valence_start_datetime = datetime.fromtimestamp(valence_ts_buffer[0])
                    valence_end_datetime = datetime.fromtimestamp(valence_ts_buffer[-1])
                    
                    self.log_predictions( 
                                        stress_start_datetime, stress_end_datetime,
                                        valence_start_datetime, valence_end_datetime,
                                        stress_pred, valence_pred)
                    
                    # Clear stress buffer and timestamps
                    self.ecg_stress_buffer.clear()
                    self.stress_timestamps.clear()
                    
                    # If valence buffer is full, clear it and its timestamps
                    if len(self.ecg_valence_buffer) == self.valence_buffer_size:
                        self.ecg_valence_buffer.clear()
                        self.valence_timestamps.clear()
            
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)  # Prevent tight error loop

def run_data_processor(mac_address):
    """
    Function to run the data processor in a thread
    """
    processor = DataProcessor(mac_address)
    processor.main_loop()

# For the server file, you can access predictions like this:
def get_latest_prediction(processor):
    """
    Retrieve the latest prediction from the queue
    """
    try:
        return processor.prediction_queue.get_nowait()
    except queue.Empty:
        return None

# Example of how to use threading
if __name__ == "__main__":
    mac_address = "98:D3:91:FD:40:9B"
    
    # Create and start the data processing thread
    data_thread = threading.Thread(target=run_data_processor, args=(mac_address,), daemon=True)
    data_thread.start()
    
    # Your server or other logic can run in the main thread
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping data processing...")

###################################################################################################
# Main Loop
###################################################################################################
'''
def main_loop():
    """
    Main loop to acquire data, process buffer when full, and log predictions.
    """
    while True:
        samples, ts = acquire_sample(inlet, ecg_buffer, timestamps)
        print(f"SAMPLES : {samples} || TS : {ts}")
        write_raw_data(ts, samples)
        
        # When buffer is full (5-second data), process and log predictions
        if len(ecg_buffer) == buffer_size:
            #stress_pred, valence_pred, _, _, ts_buffer = process_buffers(ecg_buffer, timestamps)
            #start_datetime = datetime.fromtimestamp(ts_buffer[0])
            #end_datetime = datetime.fromtimestamp(ts_buffer[-1])
            #log_predictions(output_csv, start_datetime, end_datetime, stress_pred, valence_pred)
            latest_stress, latest_valence = get_stress_valence()
            ecg_buffer.clear()
            timestamps.clear()

###################################################################################################
# Stream Initialization and Entry Point
###################################################################################################
mac_address = "98:D3:91:FD:40:9B"
print("# Looking for an available OpenSignals stream from the specified device...")
os_stream = resolve_stream("type", mac_address)
if not os_stream:
    print(f"No stream found for device with MAC address {mac_address}.")
    exit()
inlet = StreamInlet(os_stream[0])
new_session()  # Initialize raw data CSV file




if __name__ == "__main__":
    try:
        print("Streaming data... Press Ctrl+C to stop.")
        
        #asyncio.run(main())
        main_loop()
        
        
    except KeyboardInterrupt:
        print(f"Data streaming stopped and saved to {csv_file_path_ECG}")'''
