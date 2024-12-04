import pandas as pd
import numpy as np
from typing import List, Optional, Union

class RawProcessing:
    def __init__(self,
                 filename: str,
                 cols_for_activity: Union[List[str], str],
                 col_for_mets: Optional[str] = None,
                 is_emno: bool = False,
                 is_act_count: bool = False,
                 col_for_datetime: str = "time",
                 start_of_week: Union[int, str] = -1,
                 strftime: Optional[str] = None,
                 col_for_pid: Optional[str] = None,
                 pid: int = -1,
                 col_for_hr: Optional[str] = None,
                 additional_data: Optional[object] = None,
                 device_location: Optional[str] = None):
        """
        Initializes the RawProcessing object.

        Parameters
        ----------
        filename : str
            Path to the file to be analyzed.
        cols_for_activity : list or str
            Columns that record activity.
        col_for_mets : str, optional
            Column that records METs.
        is_emno : bool, optional
            True if cols_for_activity are already computed as ENMO.
        is_act_count : bool, optional
            True if cols_for_activity are already computed as counts.
        col_for_datetime : str, optional
            Name of the timestamp column. Default is "time".
        start_of_week : int or str, optional
            Day that represents the start of the week. Default is -1.
        strftime : str, optional
            Format to parse col_for_datetime.
        col_for_pid : str, optional
            Column that contains participant ID.
        pid : int, optional
            Participant ID.
        col_for_hr : str, optional
            Column that contains heart rate data.
        additional_data : object, optional
            Any additional data.
        device_location : str, optional
            Location of the device (e.g., "bw", "hip").
            
        Method 
        ------
        __configure_activity : Write the data in col_for_activity to internal_activity_cols from the beginning,
        __configure_datetime : Convert pandas.datatime object
        __configure_pid : Write the data in col_for_pid to self.pid 
        __col_fot_ht : Configure HR of rawdata to internal_hr_col 
        """
        self.filename = filename
        self.device_location = device_location
        self.additional_data = additional_data

        self.internal_activity_cols = ["hyp_act_x", "hyp_act_y", "hyp_act_z"]
        self.internal_time_col = "hyp_time_col"
        self.internal_mets_col = None
        self.naxis = 0
        self.is_act_count = False
        self.is_emno = False
        self.pid = None

        self.data = self.__load_wearable_data(self.filename)
        self.__configure_activity(cols_for_activity, col_for_mets, is_emno, is_act_count) 
        self.__configure_datetime(col_for_datetime, strftime, start_of_week)
        self.__configure_pid(col_for_pid, pid)
        self.__configure_hr(col_for_hr)

    def get_pid(self) -> Optional[int]:
        return self.pid

    def set_time_col(self, new_name: str):
        if new_name:
            self.internal_time_col = new_name

    def __configure_hr(self, col_for_hr: Optional[str]):

        self.internal_hr_col = col_for_hr

    def __configure_activity(self, cols_for_activity: Union[List[str], str], col_for_mets: Optional[str], is_emno: bool, is_act_count: bool):

        self.is_act_count = is_act_count
        self.is_emno = is_emno
        self.naxis = len(cols_for_activity) if isinstance(cols_for_activity, list) else 1

        if self.naxis == 0:
            raise ValueError("Need at least one column to represent activity.")

        if self.naxis > 3:
            raise ValueError("Current implementation allows up to 3 columns for physical activity.")

        if col_for_mets:
            self.internal_mets_col = col_for_mets

        if isinstance(cols_for_activity, str):
            cols_for_activity = [cols_for_activity]

        for i, col in enumerate(cols_for_activity):
            if col not in self.data.columns:
                raise ValueError(f"Column {col} not detected in the dataset. Available columns are {', '.join(self.data.columns)}")
            self.data[self.internal_activity_cols[i]] = self.data[col]

    def __configure_pid(self, col_for_pid: Optional[str], pid: int):
        if col_for_pid is None and pid == -1:
            raise ValueError("Either pid or col_for_pid need to have a valid value.")

        if pid != -1:
            self.pid = pid
        elif col_for_pid:
            if col_for_pid not in self.data.columns:
                raise ValueError(f"Column {col_for_pid} is not in the dataframe.")
            self.pid = self.data[col_for_pid].iloc[0]

    def __configure_datetime(self, col_for_datetime: str, strftime: Optional[str], start_of_week: Union[int, str]):
        if strftime is None and start_of_week is None:
            raise ValueError("Either strftime or start_of_week need to have a valid value.")

        if strftime is None or "%d" not in strftime:
            starting_day_of_week = 1 if isinstance(start_of_week, int) else self.data[start_of_week].iloc[0]
            self.__datetime_without_date(col_for_datetime, starting_day_of_week)
        else:
            self.data[self.internal_time_col] = pd.to_datetime(self.data[col_for_datetime], format=strftime)

    def __datetime_without_date(self, col_for_datetime: str, starting_day_of_week: int):
        freq = abs(int(self.data[col_for_datetime].iloc[1][-2:]) - int(self.data[col_for_datetime].iloc[0][-2:]))
        ndays = int(np.ceil(self.data.shape[0] / (24 * (60 / freq) * 60)))
        first_time, last_time = self.data[col_for_datetime].iloc[0], self.data[col_for_datetime].iloc[-1]

        for n in range(-1, 2):
            times = pd.date_range(start=f"1-{starting_day_of_week}-2017 {first_time}",
                                  end=f"1-{starting_day_of_week + ndays + n}-2017 {last_time}",
                                  freq=f"{freq}s")

            if len(times) == len(self.data):
                self.data[self.internal_time_col] = times
                break
        else:
            raise ValueError(f"Could not find correct range for dataframe. Please check if parameter 'datetime_col' (={col_for_datetime}) is correct and has all its entries valid.")

    def export_hypnospy(self, filename: str):
        self.data.to_hdf(filename, key='data', mode='w')
        metadata = pd.Series([self.pid, self.internal_time_col, self.internal_activity_cols[:self.naxis], self.internal_mets_col, self.is_act_count, self.is_emno, self.device_location, self.additional_data])
        metadata.to_hdf(filename, key='other')
        print(f"Saved file {filename}.")

    def __load_wearable_data(self, filename: str) -> pd.DataFrame:
        f = filename.lower()
        if f.endswith(('.cwa', '.cwa.gz', 'CWA')):
            return self.__process_axivity(filename)
        elif f.endswith('.bin'):
            return self.__process_geneactiv(filename)
        elif f.endswith('.dat'):
            return self.__process_actigraph(filename)
        elif f.endswith(('.csv', '.csv.gz')):
            return self.__process_csv(filename)
        else:
            raise ValueError(f"ERROR: Wearable format not supported for file: {filename}")

    def __process_csv(self, csvfile: str) -> pd.DataFrame:
        return pd.read_csv(csvfile)

    def __process_axivity(self, cwaFile: str) -> pd.DataFrame:
        # Add your specific processing code for Axivity devices here
        pass

    def __process_actigraph(self, datFile: str) -> pd.DataFrame:
        # Add your specific processing code for Actigraph devices here
        pass

    def __process_geneactiv(self, datFile: str) -> pd.DataFrame:
        # Add your specific processing code for Geneactiv devices here
        pass

    def run_nonwear(self):
        # Implement the nonwear detection using PAMPRO here
        pass

    def calibrate_data(self):
        # Implement the data calibration using PAMPRO here
        pass

    def obtain_PA_metrics(self):
        # Implement the derivation of PA metrics using PAMPRO here
        pass

class ActiwatchSleepData(RawProcessing):
    """ RawProcessing child class to be used when working with Actiwatch data
    """
        
        
    def __init__(self, filename, device_location=None, col_for_datetime="time", col_for_pid="pid"):
        """
        

        Parameters
        ----------
        Specific for files from Actiwatch devices. See RawProcessing() documentation for further info.

        Returns
        -------
        None.

        """
        super().__init__(filename, device_location=device_location,
                         cols_for_activity=["activity"],
                         is_act_count=True,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         start_of_week="dayofweek",
                         # Participant information
                         col_for_pid=col_for_pid,
                         )
        self.device = "actigraphy"
        self.data["hyp_annotation"] = self.data["interval"].isin(["REST", "REST-S"])


# TODO: missing Actiheart (Fenland, BBVS), Axivity (BBVS, Biobank)

class MESAPreProcessing(RawProcessing):
    """ RawProcessing child class to be used when working with data from the MESA study
       https://www.mesa-nhlbi.org/
    
    """

    def __init__(self, filename, device_location=None, col_for_datetime="linetime", col_for_pid="mesaid"):
        """
        

        Parameters
        ----------
        Specific for files from the MESA Study. See RawProcessing() documentation for further info.

        Returns
        -------
        None.

        """
        super().__init__(filename, device_location=device_location,
                         cols_for_activity=["activity"],
                         is_act_count=True,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         start_of_week="dayofweek",
                         # Participant information
                         col_for_pid=col_for_pid,
                         )
        self.device = "actigraphy"
        self.data["hyp_annotation"] = self.data["interval"].isin(["REST", "REST-S"])


class MMASHPreProcessing(RawProcessing):
    """ RawProcessing child class to be used when working with data from the MMASH dataset
       https://physionet.org/content/mmash/1.0.0/
    
    """

    def __init__(self, filename, device_location=None, col_for_datetime="time", col_for_pid="pid",
                 col_for_hr="HR", cols_for_activity=["Axis1", "Axis2", "Axis3"], strftime="%Y-%b-%d %H:%M:%S"):
        """
        

        Parameters
        ----------
        Specific for files from the MMASH dataset. See RawProcessing() documentation for further info.

        Returns
        -------
        None.

        """
        super().__init__(filename, device_location=device_location,
                         # Activity information
                         cols_for_activity=cols_for_activity,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         strftime=strftime,
                         # Participant information
                         col_for_pid=col_for_pid,
                         # HR information:
                         col_for_hr=col_for_hr,
                         )
        self.device = "actigraphy"

class HCHSPreProcessing(RawProcessing):
    """RawProcessing child class to be used when working with data from the MMASH dataset
       https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000810.v1.p1
    
    """

    def __init__(self, filename, device_location=None, col_for_datetime="time", col_for_pid="pid"):
        """
        

        Parameters
        ----------
        Specific for files from the HCHS dataset. See RawProcessing() documentation for further info.

        Returns
        -------
        None.

        """
        super().__init__(filename, device_location=device_location,
                         cols_for_activity=["activity"],
                         is_act_count=True,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         start_of_week="dayofweek",
                         # Participant information
                         col_for_pid=col_for_pid,
                         )
        self.device = "actigraphy"
        self.data["hyp_annotation"] = self.data["interval"].isin(["REST", "REST-S"])
