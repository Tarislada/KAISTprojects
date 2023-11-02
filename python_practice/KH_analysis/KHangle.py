import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class Preprocess:
    def __init__(self, data_path):
        """
        Initialize the Preprecessing object

        Parameters:
            data_path (str): Path to the CSV data file.
        """      
        # Load raw data
        self.df = pd.read_csv(data_path)
        self.output_dfs = []
    
    def body_centered(self,target_column_x='nose x',target_column_y='nose y',center_column_x='body center x',center_column_y='body center y'):
        """
        Recenter the body coordinate to origin - all locomotion canceled out
        Output in a 2D vector form 

        Parameters: 
            target_column_x (str): column describing x coordinate of the target
            target_column_y (str): column describing y coordinate of the target
            center_column_x (str): column describing x coordinate of the to-be origin
            center_column_y (str): column describing y coordinate of the to-be origin 
        """
        # Center data
        self.df['vector_x'] = self.df[target_column_x]-self.df[center_column_x]
        self.df['vector_y'] = self.df[target_column_y]-self.df[center_column_y]

        return self

    def separate_frames(self, expected_num=2, judgement_col='vector_x'):
        """
        Separate the rows based on the specified judgement column value.
        
        Rows with the top N values (where N=expected_num) for the judgement column among duplicates are stored in separate dataframes.
        
        :param expected_num: Expected number of rows for a single frame.
        :param judgement_col: Column name based on which the separation decision will be made.
        """
        # Sort the dataframe by frame number and the judgement column value
        sorted_df = self.df.sort_values(by=['frame', judgement_col], ascending=[True, False])
        
        # Separate based on the judgement column value
        for i in range(expected_num):
            current_df = sorted_df.drop_duplicates(subset='frame', keep='first')
            self.output_dfs.append(current_df) 
            sorted_df = sorted_df[~sorted_df.index.isin(current_df.index)]
            
        # Identify frames with more rows than expected
        frame_counts = self.df['frame'].value_counts()
        excess_frames = frame_counts[frame_counts > expected_num].index
        
        # Print warning for frames with more than the expected number of rows and discard the excess rows
        for frame in excess_frames:
            print(f"Warning: Frame {frame} has more than {expected_num} rows. Discarding the excess rows.")
        
        return self 
    
    def seperate_conditions(self,cut_offs,target_df=0):
        """
        Cut File row-wise (frame-wise) to seperate conditions
        
        Parameter:
            cutoffs (int): row-wise cutoff number
            target_df (int): the index df to be splited. default=0
        """
        if target_df >= len(self.output_dfs) or target_df < 0:
            raise ValueError("Invalid dataframe index.")
        
        if not isinstance(cut_offs, list):
            cut_offs = [cut_offs]
        
        self.dfs = []
        df_to_split = self.output_dfs[target_df]
        
        # Add a starting point and an ending point for slicing
        cut_points = [0] + sorted(cut_offs) + [len(df_to_split)]
        
        for i in range(len(cut_points) - 1):
            start, end = cut_points[i], cut_points[i+1]
            self.dfs.append(df_to_split.iloc[start:end])
               
        return self

    def save_seperate_mouse_csv(self, filenames):
        """
        Save the dataframes to separate CSV files.
        
        :param filenames: List of filenames for the CSVs.
        """
        for i, filename in enumerate(filenames):
            self.output_dfs[i].to_csv(filename,index=False)
            # df_key = f"df_rank_{i+1}"
            # if df_key in self.output_dfs:
            #     self.output_dfs[df_key].to_csv(filename, index=False)
        return self
    
    def save_conditional_csv(self,filenames):
        if len(self.dfs) != len(filenames):
            raise ValueError("number of filenames must match number of splits")
        
        for i in range(len(self.dfs)):
            self.dfs[i].to_csv(filenames[i]+'.csv')

class PolarHeatmap:
    def __init__(self, data_path, bin_size=15, num_theta_bins=36, num_r_bins=10):
        """
        Initialize the PolarHeatmap object.
        
        Parameters:
            data_path (str): Path to the CSV data file.
            bin_size (int): Number of frames per bin for angle and length adjustment.
            num_theta_bins (int): Number of bins in the angular direction for the heatmap.
            num_r_bins (int): Number of bins in the radial direction for the heatmap.
        """
        # Load the data and set binning parameters
        self.data = pd.read_csv(data_path)
        self.bin_size = bin_size
        self.num_theta_bins = num_theta_bins
        self.num_r_bins = num_r_bins
        
        # Perform initial calculations and adjustments
        self._calculate_original_vectors()
        self._adjust_vectors()

    def _calculate_original_vectors(self):
        """
        Calculate the original angles and lengths for each vector in the dataset with np.arctan2.
        """
        # Calculating the original angles and lengths from the x, y vectors
        self.data['original_angle_rad'] = np.arctan2(self.data['vector_y'], self.data['vector_x'])
        self.data['original_length'] = np.sqrt(self.data['vector_x']**2 + self.data['vector_y']**2)

        return self
        
    def _adjust_vectors(self):
        """
        Adjust the angles and lengths based on specifications:
        - Angles are adjusted by subtracting the reference angle in each bin.
        - Lengths are replaced by the mean length within each bin.
        - Max angles are calculated by collecting and echoing the max value of each bin
        """
        # Grouping the data into bins
        grouped = self.data.groupby(self.data['frame'] // self.bin_size)
        
        # Lists to store the adjusted angles and lengths
        adjusted_angles = []
        adjusted_lengths = []
        max_angles = []

        # Loop through each bin/group, and adjust the angles and lengths
        for _, group in grouped:
            # Reference angle is the angle of the first frame in the bin
            reference_angle = group.iloc[0]['original_angle_rad']
            # Mean length is calculated from all frames in the bin
            mean_length = group['original_length'].mean()
    
            # Adjust angles by subtracting the reference angle and lengths by replacing with mean length
            adjusted_angles_bin = group['original_angle_rad'] - reference_angle
            max_angle_bin = [abs(adjusted_angles_bin).max()] * len(group)
            adjusted_lengths_bin = [mean_length] * len(group)
    
            # Add the adjusted angles and lengths to the respective lists
            adjusted_angles.extend(adjusted_angles_bin)
            adjusted_lengths.extend(adjusted_lengths_bin)
            max_angles.extend(max_angle_bin)

        # Add the adjusted angles and lengths to the DataFrame
        self.data['max_angle_rad'] = max_angles
        self.data['adjusted_angle_rad'] = adjusted_angles
        self.data['adjusted_length'] = adjusted_lengths

        return self
        
    def plot_heatmap(self, cmap='jet',angle_column_name='adjusted_angle_rad',length_column_name='adjusted_length'):
        """
        Generate a heatmap-style polar plot.
        
        Parameters:
            cmap (str): Colormap for the heatmap.
            angle_column_name (str): Column name for angle data to plot
            length_column_name (str): Column name for length data to plot
        """
        # Define the bin edges for the histogram
        theta_bins = np.linspace(-np.pi, np.pi, self.num_theta_bins + 1)
        r_bins = np.linspace(self.data[length_column_name].min(), 
                             self.data[length_column_name].max(), 
                             self.num_r_bins + 1)

        # Calculate the 2D histogram
        hist, _, _, _ = binned_statistic_2d(
            x=self.data[angle_column_name], 
            y=self.data[length_column_name], 
            values=None, statistic='count', 
            bins=[theta_bins, r_bins]
        )

        # Generate the polar plot and display the heatmap
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection='polar')
        
        ax.set_title('Heatmap-Style Polar Plot', va='bottom')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(90)
        pc = ax.pcolormesh(theta_bins, r_bins, hist.T, cmap=cmap, shading='auto')
        
        # Add a colorbar and labels
        plt.colorbar(pc, label='Frequency', ax=ax)
        
        plt.grid(True)
        plt.show()
        return self

    def save_adjusted_data(self, output_path): # to be changed to self.data.to_csv(output_path,index=False)
        """
        Save the adjusted data to a CSV file.
        
        Parameters:
            output_path (str): Path for the output CSV file.
        """
        # Selecting relevant columns and saving them to a CSV
        # self.data[['frame', 'vector_x', 'vector_y', 'original_angle_rad', 'original_length', 
                #    'adjusted_angle_rad', 'adjusted_length']].to_csv(output_path, index=False)
        self.data.to_csv(output_path,index=False)
        return self

    def save_pre_reference_rows(self, output_path):
            """
            Save rows just before the reference rows in each bin to a CSV file.
            
            Parameters:
                output_path (str): Path for the output CSV file.
            """
            # Get the index of rows just before the reference rows in each bin
            pre_reference_rows_idx = np.arange(self.bin_size - 1, len(self.data), self.bin_size)
            
            # Extract rows using the indices
            pre_reference_rows = self.data.iloc[pre_reference_rows_idx]
            
            # Save the selected rows to a CSV
            pre_reference_rows[['frame', 'vector_x', 'vector_y', 'original_angle_rad', 'original_length', 
                                'adjusted_angle_rad', 'adjusted_length']].to_csv(output_path, index=False)
            return self


# Usage example:
# preprocess = Preprocess(r'C:\Users\endyd\OneDrive\문서\GitHub\python_practice\KH_analysis\KH_m23_24_1mw_20Hz_target_columns2.csv')
# preprocess.body_centered().separate_frames()
# preprocess.seperate_conditions([5700,11100]).save_conditional_csv(['Pre_off','ON','Post_off'])
# preprocess.save_seperate_mouse_csv((r'C:\Users\endyd\OneDrive\문서\GitHub\python_practice\KH_analysis\m23_1mw_20Hz.csv',r'C:\Users\endyd\OneDrive\문서\GitHub\python_practice\KH_analysis\m24_1mw_20Hz.csv'))
polar_heatmap = PolarHeatmap('KH_analysis/ON.csv',bin_size=90,num_theta_bins=120)
polar_heatmap.plot_heatmap(cmap='jet')
# polar_heatmap.plot_heatmap(cmap='jet',angle_column_name='max_angle_rad')
# polar_heatmap.save_adjusted_data('adjusted_data.csv')
polar_heatmap.save_pre_reference_rows('adjusted_data.csv')