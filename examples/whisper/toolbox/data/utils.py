import numpy as np
from tqdm import tqdm
from naludaq.tools.pedestals import PedestalsCorrecter
from naluacq import Acquisition
from numpy.lib.format import open_memmap

def correct_data(acq, peds,correct_in_place=False):
    corrected_data = []
    errors = []
    num_channels, num_samples = None, None
    for idx in tqdm(range(len(acq)), desc="Applying pedestal corrections", total=len(acq)):
        try:
            corrector = PedestalsCorrecter(acq.params, peds)
            if num_channels is None or num_samples is None:
                num_channels, num_samples = np.array(acq[idx]["data"]).shape
            corrected_data.append(corrector.run(acq[idx], correct_in_place=correct_in_place))
        except Exception as e:
            corrected_data.append(None)
            errors.append(f"IDX {idx}: {e}")

    if num_channels is not None and num_samples is not None:
        for i, item in enumerate(corrected_data):
            if item is None:
                corrected_data[i] = {"data": np.full((num_channels, num_samples), np.nan)}
    if errors:
        print("Errors for:")
        for error in errors:
            print(error)
    return corrected_data

def extract_data(acq):
    data = []
    for idx in tqdm(range(len(acq)), desc="Extracting data"):
        try: 
            data.append(acq[idx]["data"])
        except:
            pass
    return np.array(data)

def get_event(index: int, acq: Acquisition = None, pedestals: np.array = None, dir: str = None, apply_pedestals: bool =True) -> np.array:
    """
    _summary_

    Args:
        index (int): Index of the event
        acq (DiskAcquisition, optional): The DiskAcquisition object containing the dataset. Defaults to None.
        pedestals (np.array, optional): The pedestals for the dataset. Defaults to None.
        dir (str, optional): The directory of the acquistion data that should be loaded. Defaults to None.
        apply_pedestals (bool, optional): Applies pedestals. Defaults to True.

    Returns:
        np.array: A numpy array of dimensions (# of channels, # of samples)
    """
    event = None
    params = None
    num_acquisitions = 0

    pedestals = acq.pedestals_calibration if pedestals == None else pedestals

    if acq is not None:
        event = acq[index]
        params = acq.params
        num_acquisitions = len(acq)
    else:
        with Acquisition(dir) as acq:
            event = acq[index]
            params = acq.params
    
    assert index < num_acquisitions - 1, f"Index must be less than {num_acquisitions - 1}"
        
    if apply_pedestals:
        if pedestals is None:
            print("Acquisition must have pedestals in order to correct data!")
        else: 
            # Apply correction (in place)
            corrector = PedestalsCorrecter(params, pedestals)
            corrected_event = corrector.run(event, correct_in_place=True)
            return corrected_event
        
    else:
        return event
    
    def build_chip_memmap(data, indices, chip_slice, out_name, channels=4, mm_dir = "mm_dir"):
        path = mm_dir / out_name

        shape = (len(indices), channels, data.shape[-1])
        mm = open_memmap(path, mode="w+", dtype=data.dtype, shape=shape)
        chunk = 1024
        for start in range(0, len(indices), chunk):
            stop = min(start + chunk, len(indices))
            idx = indices[start:stop]
            mm[start:stop] = data[idx, chip_slice, :]
        mm.flush()
        return mm