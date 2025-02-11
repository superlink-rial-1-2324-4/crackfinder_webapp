import re
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import zipfile

from datetime import datetime
from scipy.signal import savgol_filter
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely import buffer
from simplification.cutil import simplify_coords
from sklearn.cluster import DBSCAN

import cv2
from ultralytics import YOLO

# preferences
SHOW_VISUALIZATION = False
SCALE_YAW = 1
THRESHOLD_YAW = 0.25

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        execution_time = end_time - start_time
        # ANSI escape codes for green and bold text
        print(f"\n\033[1;32mFunction '{func.__name__}' executed in {execution_time:.4f} seconds\033[0m.\n")
        return result
    return wrapper

@measure_time
def DataPreparation():

    # load data from uploads
    mycsv = ''
    for filename in os.listdir('uploads'):
        if filename.lower().endswith('.csv'):
            mycsv = filename
    mycsv = os.path.join('uploads', mycsv)
    df_flowdeckdata = pd.read_csv(mycsv)
    x_coords = df_flowdeckdata['Y'].tolist()
    y_coords = df_flowdeckdata['X'].tolist()
    z_coords = df_flowdeckdata['Z'].tolist()
    yaws = df_flowdeckdata['Yaw'].tolist()
    timestamps = df_flowdeckdata['Timestamp'].tolist()

    # preprocessing of data
    x_coords = [-x for x in x_coords]
    yaws = [SCALE_YAW*yaw for yaw in yaws]
    minz = min(z_coords)
    z_coords = [z-minz for z in z_coords]

    # add dead time to the beginning
    x_coords = [0] * 200 + x_coords
    y_coords = [0] * 200 + y_coords
    z_coords = [0] * 200 + z_coords
    yaws = [0] * 200 + yaws
    timestamps = ['25/09/2024  12:17:12 pm'] * 200 + timestamps

    # update flowdeck data
    df_flowdeckdata_updated = pd.DataFrame()
    df_flowdeckdata_updated['X'] = x_coords
    df_flowdeckdata_updated['Y'] = y_coords
    df_flowdeckdata_updated['Z'] = z_coords
    df_flowdeckdata_updated['Yaw'] = yaws
    df_flowdeckdata_updated['Timestamp'] = timestamps
    
    # determine filename of zip file
    myzip = ''
    for filename in os.listdir('uploads'):
        if filename.lower().endswith('.zip'):
            myzip = filename
    myzip = os.path.join('uploads', myzip)

    # preallocate for image data dataframe
    df_imagedata = pd.DataFrame()

    # extract images and times of capture
    with zipfile.ZipFile(myzip, 'r') as file:
        # pre allocate for filenames and datetime
        filenames = [''] * len(file.infolist())
        filetimes = [''] * len(file.infolist())
        for i, file_info in enumerate(file.infolist()):
            # check if file is an image file
            image_extensions = {'.dng', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
            isimage = False
            for extension in image_extensions:
                if file_info.filename.lower().endswith(extension):
                    isimage = True
                    break
            if isimage:
                # get filename
                filenames[i] = file_info.filename
                # get datetime
                filetimes[i] = datetime.fromtimestamp(time.mktime(file_info.date_time + (0, 0, 0))).strftime("%Y-%m-%d %H:%M:%S").replace(" 0", " ")

        # store image filename and datetime in dataframe
        df_imagedata = pd.DataFrame({'ImagePath': filenames, 'Timestamp': filetimes})
        df_imagedata.replace('', np.nan, inplace=True)
        df_imagedata = df_imagedata.dropna().reset_index(drop=True)
    
    # get the date of the survey
    dates = df_imagedata['Timestamp'].tolist()
    dt = datetime.strptime(dates[0], "%Y-%m-%d %H:%M:%S")
    year = str(dt.year)
    month = str(dt.month)
    day = str(dt.day)
    if len(month) == 1: month = '0' + month
    if len(day) == 1: day = '0' + day
    index = str(sum(1 for file in os.listdir('static') if file.startswith(f"rawloc_{year}_{month}_{day}")))

    # generate session id
    id = f"{year}_{month}_{day}_{index}"

    # update location data csv
    df_flowdeckdata_updated.to_csv(os.path.join('static', f'rawloc_{id}.csv'), index=False)
    
    # create csv containing image metadata
    df_imagedata.to_csv(os.path.join('static', f'imagedata_{id}.csv'), index=False)
    
    # extract zip image contents
    with zipfile.ZipFile(myzip, 'r') as file:
        os.mkdir(os.path.join('static', f"images_{id}"))
        file.extractall(os.path.join('static', f"images_{id}"))
    
    # clean uploads folder
    #os.remove(myzip)
    #os.remove(mycsv)

    return id

@measure_time
def YawTransform(id):
    
    # load flowdeck data
    df_flowdeckdata = pd.read_csv(os.path.join('static', f'rawloc_{id}.csv'))
    x_coords = df_flowdeckdata['X']
    y_coords = df_flowdeckdata['Y']
    z_coords = df_flowdeckdata['Z']
    yaws = df_flowdeckdata['Yaw']
    timestamps = df_flowdeckdata['Timestamp']

    # for runs with no turns
    
    no_turn = True
    for yaw in yaws:
        if abs(yaw) > 30:
            no_turn = False
            break
    if no_turn:
        df_yawtrans = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'Z': z_coords,
            'Yaw': yaws,
            'Timestamp': timestamps
        })
        df_yawtrans.to_csv(os.path.join('static', f'yawtrans_{id}.csv'), index=False)
        return

    # smoothen data
    x_coords = savgol_filter(x_coords, window_length=11, polyorder=10)
    y_coords = savgol_filter(y_coords, window_length=11, polyorder=10)
    z_coords = savgol_filter(z_coords, window_length=11, polyorder=10)
    yaws = savgol_filter(yaws, window_length=11, polyorder=10)

    # detect positive/negative peak yaw values and indices
    yaw_absolute = [abs(yaw) for yaw in yaws]
    yaw_diff = np.diff(yaw_absolute)
    yaw_diff = np.convolve(yaw_diff / np.max(yaw_diff), np.ones(3)/3, mode='valid')
    yaw_peaks_indices = np.where(yaw_diff > THRESHOLD_YAW)[0]

    # clean the peak indices
    temp = []
    for i, val in enumerate(np.diff(yaw_peaks_indices)):
        if abs(val) > 1:
            temp.append(yaw_peaks_indices[i-1])
    temp.append(yaw_peaks_indices[-1])
    yaw_peaks_indices = np.array(temp) + 5

    # retain yaw values at peaks
    yaw_retained = [0] * len(yaws)
    for i in range(1, len(yaws)):
        if i in yaw_peaks_indices:
            yaw_retained[i] = yaw_retained[i-1] + yaws[i]
        else:
            yaw_retained[i] = yaw_retained[i-1]

    # convert yaw in degrees to radians
    yaws_rad = np.radians(yaw_retained)

    # initializations for transformation
    prevx, prevy, vx, vy = 0, 0, 0, 0
    x_coords_transformed, y_coords_transformed  = [], []

    # rotation of coordinates
    for x, y, yaw in zip(x_coords, y_coords, yaws_rad):
        
        # calculate the new values of x and y coordinates
        x_new = round(vx + (x-prevx) * np.cos(yaw) - (y-prevy) * np.sin(yaw), 3)
        y_new = round(vy + (x-prevx) * np.sin(yaw) + (y-prevy) * np.cos(yaw), 3)
        
        # set the vertex and shifting factors for the next iteration
        prevx, prevy = x, y
        vx, vy = x_new, y_new
        
        # record the results
        x_coords_transformed.append(x_new)
        y_coords_transformed.append(y_new)
    
    # write the rotated coordinates and
    # retained yaw values to new csv
    df_yawtrans = pd.DataFrame({
        'X': x_coords_transformed,
        'Y': y_coords_transformed,
        'Z': z_coords,
        'Yaw': yaw_retained,
        'Timestamp': timestamps
    })
    df_yawtrans.to_csv(os.path.join('static', f'yawtrans_{id}.csv'), index=False)
    
    # visualization
    if SHOW_VISUALIZATION: 
        plt.plot(yaws, '-', label='Data')
        plt.plot(yaw_retained, '-', label='Retained')
        plt.scatter(yaw_peaks_indices, [yaws[i] for i in yaw_peaks_indices], color='red', label='Peaks', zorder=5)
        plt.legend()
        plt.show()

        plt.plot(yaws / np.max(abs(yaws)), '-', label='Original')
        plt.plot(yaw_diff / np.max(abs(yaw_diff)), '-', label='Diff')
        plt.scatter(yaw_peaks_indices, [(yaw_diff / np.max(abs(yaw_diff)))[i] for i in yaw_peaks_indices], color='red', label='Peaks', zorder=5)
        plt.legend()
        plt.show()

        plt.plot(x_coords, y_coords, label='Original')
        plt.plot(x_coords_transformed, y_coords_transformed, label='Transformed')
        plt.legend()
        plt.grid('on')
        plt.show()

    return

@measure_time
def MapGenerate(id):

    def smooth_coordinates(x, y, window_size=11, poly_order=2):
        # Ensure the window size is an odd integer
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer")
        
        # Apply Savitzky-Golay filter to smooth both x and y coordinates
        x_smooth = savgol_filter(x, window_length=window_size, polyorder=poly_order)
        y_smooth = savgol_filter(y, window_length=window_size, polyorder=poly_order)
        
        return x_smooth, y_smooth

    def calculate_centroid(x_coords, y_coords):

        if len(x_coords) != len(y_coords):
            raise ValueError("The number of x-coordinates and y-coordinates must be the same.")
        
        # Calculate the averages
        x_centroid = sum(x_coords) / len(x_coords)
        y_centroid = sum(y_coords) / len(y_coords)

        return x_centroid, y_centroid

    def translate_coordinates(x_coords, y_coords, x_centroid, y_centroid):
        xc, yc = calculate_centroid(x_coords, y_coords)
        x_shift = x_centroid - xc
        y_shift = y_centroid - yc
        return [x + x_shift for x in x_coords], [y + y_shift for y in y_coords]

    def simplify_path(x, y, tolerance=1.0):
        coords = np.column_stack((x, y))
        simplified = simplify_coords(coords, tolerance)
        return simplified[:, 0], simplified[:, 1]

    def detect_corners_angle(x, y, min_angle_threshold=45, max_angle_threshold=120, min_distance=1.0):
        """
        Detect corners based on angle between consecutive segments and omit corners
        that are too close to the last detected corner.

        Parameters:
        x (list or np.array): Smoothed x coordinates.
        y (list or np.array): Smoothed y coordinates.
        angle_threshold (float): Angle change threshold in degrees to detect corners.
        min_distance (float): Minimum distance between consecutive detected corners.

        Returns:
        x_corners, y_corners (list of floats): Detected corners' x and y coordinates.
        """
        x_corners, y_corners = [], []

        for i in range(1, len(x) - 1):
            # Vectors between three consecutive points
            v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
            v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
            
            # Normalize the vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Compute the angle between vectors using dot product
            angle = np.degrees(np.arccos(np.dot(v1_norm, v2_norm)))
            
            # Detect if angle exceeds threshold
            if max_angle_threshold > angle > min_angle_threshold:
                
                # If no corners have been detected yet, add the first one
                if len(x_corners) == 0:
                    x_corners.append(x[i])
                    y_corners.append(y[i])
                else:
                    # Calculate the distance from the last detected corner
                    dist = np.sqrt((x[i] - x_corners[-1])**2 + (y[i] - y_corners[-1])**2)
                    
                    # Only add corner if it's farther than the min_distance
                    if dist > min_distance:
                        x_corners.append(x[i])
                        y_corners.append(y[i])
        
        return x_corners, y_corners

    def rotate_points(points, angle, center):
        """Rotate points by a given angle around a center point."""
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        centered_points = points - center
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        return rotated_points + center

    def find_bottom_base(points):
        """Find the two points that form the bottom base of a rectangle."""
        # Sort points by their y-coordinate (smallest y-values should form the bottom base)
        sorted_points = sorted(points, key=lambda point: point[1])
        return sorted_points[0], sorted_points[1]

    def align_rectangle_to_unit_square(x_coords, y_coords):
        """Align a slanted rectangle's bottom base to match the unit square's bottom base."""
        # Combine the x and y coordinates into a set of points
        rectangle_points = np.array(list(zip(x_coords, y_coords)))

        # Find the bottom base of the slanted rectangle
        bottom_left, bottom_right = find_bottom_base(rectangle_points)

        # Calculate the angle between the bottom base and the x-axis
        dx = bottom_right[0] - bottom_left[0]
        dy = bottom_right[1] - bottom_left[1]
        angle = np.arctan2(dy, dx)  # Angle between the bottom base and the x-axis

        # Calculate the center of the rectangle for rotation purposes
        center = np.mean(rectangle_points, axis=0)

        # Rotate the rectangle points by -angle to align the bottom base with the x-axis
        aligned_points = rotate_points(rectangle_points, angle, center)

        # Output separate lists of x and y coordinates of the aligned rectangle
        aligned_x = aligned_points[:, 0].tolist()
        aligned_y = aligned_points[:, 1].tolist()

        aligned_x.append(aligned_x[0])
        aligned_y.append(aligned_y[0])
        
        return aligned_x, aligned_y, angle
    
    def get_extreme_corners(x_coords, y_coords):
        """Get the upper-rightmost, upper-leftmost, lower-rightmost, and lower-leftmost points.
        Return separate lists of x and y coordinates in proper order for plotting a rectangle."""
        
        # Combine x and y coordinates into an array of points
        points = np.array(list(zip(x_coords, y_coords)))
        
        # Sort the points based on their y-coordinates
        sorted_points = sorted(points, key=lambda p: p[1])
        
        # Split into two halves: top half and bottom half
        bottom_half = sorted_points[:2]  # Points with smaller y-values
        top_half = sorted_points[2:]     # Points with larger y-values
        
        # Among the top half, find the leftmost and rightmost points
        upper_leftmost = min(top_half, key=lambda p: p[0])
        upper_rightmost = max(top_half, key=lambda p: p[0])
        
        # Among the bottom half, find the leftmost and rightmost points
        lower_leftmost = min(bottom_half, key=lambda p: p[0])
        lower_rightmost = max(bottom_half, key=lambda p: p[0])
        
        # Properly ordered coordinates for plotting a rectangle
        # Start with lower-left -> lower-right -> upper-right -> upper-left
        ordered_x = [lower_leftmost[0], lower_rightmost[0], upper_rightmost[0], upper_leftmost[0], lower_leftmost[0]]
        ordered_y = [lower_leftmost[1], lower_rightmost[1], upper_rightmost[1], upper_leftmost[1], lower_leftmost[1]]
        
        return ordered_x, ordered_y
    
    def perfect_shape(x, y):
        # determine the lower leftmost point
        bottom_left_x = min(x)
        bottom_left_y = min(y)

        # determine the longest width and length
        x_length = abs(max(x)-min(x))
        y_length = abs(max(y)-min(y))

        # form a perfect square
        x_perfect = [0, 0, 1, 1, 0]
        y_perfect = [0, 1, 1, 0, 0]

        # apply the lengths
        x_perfect = [x_length*x for x in x_perfect]
        y_perfect = [y_length*y for y in y_perfect]

        # shift the shape to the lower left most point
        x_perfect = [x+bottom_left_x for x in x_perfect]
        y_perfect = [y+bottom_left_y for y in y_perfect]

        return x_perfect, y_perfect
        
    def thicken_shape(x, y, thickness=1.0, join_style=2):
        """
        Thicken a shape defined by x and y coordinates by a specified thickness.

        Parameters:
        x (list): List of x coordinates.
        y (list): List of y coordinates.
        thickness (float): Thickness to expand the shape outward.
        join_style (int): Join style for corners (1 for bevel, 2 for mitre, 3 for round).

        Returns:
        tuple: New x and y coordinates of the thickened shape.
        """
        # Create a polygon from the coordinates
        original_shape = Polygon(zip(x, y))

        # Ensure the polygon is oriented counter-clockwise
        original_shape = orient(original_shape, sign=1.0)

        # Make the shape thicker by the specified amount with the given join style
        thicker_shape = buffer(original_shape, thickness, join_style=join_style)

        # Extract the new coordinates
        thicker_x, thicker_y = thicker_shape.exterior.xy

        return thicker_x, thicker_y

    def insert_points(x_coords, y_coords):

        # Insert more points between each corner of the expanded
        super_x_coords, super_y_coords = [], []
        for i in range(len(list(x_coords))-1):
            xpoints = round(abs(x_coords[i] - x_coords[i+1])/0.05)
            ypoints = round(abs(y_coords[i] - y_coords[i+1])/0.05)
            if xpoints>0:
                x_inserts = np.linspace(x_coords[i], x_coords[i+1], num=xpoints)
                y_inserts = np.linspace(y_coords[i], y_coords[i+1], num=xpoints)
            elif ypoints>0:
                x_inserts = np.linspace(x_coords[i], x_coords[i+1], num=ypoints)
                y_inserts = np.linspace(y_coords[i], y_coords[i+1], num=ypoints)
            for x, y in zip(x_inserts, y_inserts):
                super_x_coords.append(x)
                super_y_coords.append(y)
        return super_x_coords, super_y_coords

    # load yawtrans data
    df_flowdeckdata = pd.read_csv(os.path.join('static', f'yawtrans_{id}.csv'))
    x_coords = df_flowdeckdata['X']
    y_coords = df_flowdeckdata['Y']

    # smoothen data
    x_smooth, y_smooth = smooth_coordinates(x_coords, y_coords, window_size=5, poly_order=2)
    df_flowdeckdata['X'] = x_smooth
    df_flowdeckdata['Y'] = y_smooth
    df_flowdeckdata.to_csv(os.path.join('static', f'yawtrans_{id}.csv'), index=False)

    # simplify flight path
    x_simple, y_simple = simplify_path(x_smooth, y_smooth)

    # determine centroid of simplified flight path
    x_centroid, y_centroid = calculate_centroid(x_simple, y_simple)

    # detect corners
    x_corners, y_corners = detect_corners_angle(x_simple, y_simple)
    x_corners, y_corners = translate_coordinates(x_corners, y_corners, x_centroid, y_centroid)

    # align the shape
    x_align, y_align, angle = align_rectangle_to_unit_square(x_corners, y_corners)
    
    # get extreme corners
    x_extreme, y_extreme = get_extreme_corners(x_align, y_align)

    # perfect the rectangle
    x_perfect, y_perfect = perfect_shape(x_extreme, y_extreme)

    # thicken the shape
    x_thick, y_thick = thicken_shape(x_perfect, y_perfect)

    # insert more points between each corner
    x_room, y_room = insert_points(x_thick, y_thick)

    # visualization
    if SHOW_VISUALIZATION:
        
        plt.plot(x_coords, y_coords, '', label='YawTrans')
        plt.plot(x_smooth, y_smooth, '-', label='Smooth')
        plt.legend()
        plt.grid('on')
        plt.show()

        plt.plot(x_coords, y_coords, '', label='YawTrans')
        plt.plot(x_smooth, y_smooth, '-', label='Smooth')
        plt.plot(x_simple, y_simple, '-', label='Simple')
        plt.plot(x_corners, y_corners, 'o-', label='Estimated Flight Path')
        plt.legend()
        plt.grid('on')
        plt.show()

        plt.plot(x_simple, y_simple, '-', label='Simple')
        plt.plot(x_smooth, y_smooth, '-', label='Smooth')
        plt.plot(x_perfect, y_perfect, 'x-', label='Aligned Flight Path')
        plt.plot(x_room, y_room, '-', label='Room Trace')
        plt.legend()
        plt.grid('on')
        plt.show()
        
        plt.plot(x_corners, y_corners, 'o-', label='Estimated Flight Path')
        plt.plot(x_align, y_align, 'o-', label='Aligned')
        plt.legend()
        plt.grid('on')
        plt.show()
        
    # generation of map
    else:
        # Use the 'Agg' backend for rendering
        import matplotlib
        matplotlib.use('Agg')  

        # output image file of room tracing
        mapfilename = os.path.join('static', f'map_{id}.png')
        plt.plot(x_room, y_room)
        plt.axis('off')
        plt.fill(x_room, y_room, color='skyblue', edgecolor='black')
        plt.savefig(mapfilename, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()

    # output corner coordinates for the next subsystem
    return x_perfect, y_perfect, x_room, y_room, angle

@measure_time
def TimeMatch(id):

    # load yawtrans location data
    df_yawtransdata = pd.read_csv(os.path.join('static',  f'yawtrans_{id}.csv'))

    # load image metadata 
    df_imagedata = pd.read_csv(os.path.join('static', f'imagedata_{id}.csv'))

    one_time = []
    unique_times = df_yawtransdata['Timestamp'].unique().tolist()

    for each in unique_times:
        # filter the df by each unique timestamp
        filtered_time = df_yawtransdata[df_yawtransdata['Timestamp'] == each].copy()
        
        # calculate the mean for the filtered df
        mean_values = filtered_time.drop(columns='Timestamp').mean()
        
        # calculate the distance for the filtered DataFrame and set it using .loc
        filtered_time.loc[:, 'Distance_From_Mean'] = np.sqrt(((filtered_time.drop(columns='Timestamp') - mean_values) ** 2).sum(axis=1))
        
        # append the row with the minimum distance to the list
        closest_sample = filtered_time.loc[filtered_time['Distance_From_Mean'].idxmin()].drop(labels='Distance_From_Mean')
        one_time.append(closest_sample.tolist())

    # convert the list into a df
    df_yawtransdata = pd.DataFrame(one_time, columns=df_yawtransdata.columns)

    # match image time with raw location data
    df_surveydata = pd.merge(df_yawtransdata, df_imagedata[['ImagePath', 'Timestamp']], on='Timestamp')

    # structure dataframe to drop or include necessary columns
    df_surveydata = df_surveydata.drop('Timestamp', axis=1)
    df_surveydata['Classification'] = 'unclassified'
    df_surveydata['GridLabel'] = 0
    df_surveydata['BooleanShow'] = 1
    df_surveydata['Notes'] = '-'

    # update the image file paths
    imagepaths = df_surveydata['ImagePath'].tolist()
    for i, imagepath in enumerate(imagepaths):
        imagepaths[i] = os.path.join(f"images_{id}", imagepath).replace('\\', '/')
    df_surveydata['ImagePath'] = imagepaths

    # add row to indicate origin of flight path
    new_row = pd.DataFrame({
        "X": [0.0],
        "Y": [0.0],
        "Z": [0.0],
        "Yaw": [0.0],
        "ImagePath": ["origin"],
        "Classification": ["origin"],
        "GridLabel": [0],
        "BooleanShow": [1],
        "Notes": ["Point of flight origin."]
    })
    df_surveydata = pd.concat([df_surveydata, new_row], ignore_index=True)

    # save the time matched dataframe to csv with proper filename
    df_surveydata.to_csv(os.path.join('static', f"survey_{id}.csv"), sep='|', index=False)

    # remove unmatched images from the folder
    folder_path = os.path.join('static', f"images_{id}")
    folder_files = os.listdir(folder_path)
    for file_name in folder_files:
        if os.path.join(f"images_{id}", file_name).replace('\\', '/') not in imagepaths:
            file_path = os.path.join(folder_path, file_name)
            # Ensure it's a file before deleting
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    return

@measure_time
def GridAssign(id, x_corners, y_corners, x_room, y_room, angle):
    
    def calculate_distance(x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def calculate_centroid(x_coords, y_coords):

        if len(x_coords) != len(y_coords):
            raise ValueError("The number of x-coordinates and y-coordinates must be the same.")
        
        # Calculate the averages
        x_centroid = sum(x_coords) / len(x_coords)
        y_centroid = sum(y_coords) / len(y_coords)

        return x_centroid, y_centroid

    def translate_coordinates(x_coords, y_coords, x_shift, y_shift):
        xc, yc = calculate_centroid(x_coords, y_coords)
        return [x + x_shift for x in x_coords], [y + y_shift for y in y_coords]

    def get_nearest_grid_label(x_query, y_query, unique_x, unique_y, grid_labels):
        # Find nearest x-coordinate
        x_idx = np.searchsorted(unique_x, x_query, side="left")
        if x_idx == 0:
            nearest_x = unique_x[0]
        elif x_idx == len(unique_x):
            nearest_x = unique_x[-1]
        else:
            left_x, right_x = unique_x[x_idx - 1], unique_x[x_idx]
            nearest_x = left_x if abs(left_x - x_query) < abs(right_x - x_query) else right_x

        # Find nearest y-coordinate
        y_idx = np.searchsorted(unique_y, y_query, side="left")
        if y_idx == 0:
            nearest_y = unique_y[0]
        elif y_idx == len(unique_y):
            nearest_y = unique_y[-1]
        else:
            left_y, right_y = unique_y[y_idx - 1], unique_y[y_idx]
            nearest_y = left_y if abs(left_y - y_query) < abs(right_y - y_query) else right_y

        # Retrieve label
        return grid_labels.get((nearest_x, nearest_y), None)

    def rotate_points(x_coords, y_coords, angle, center):
        """Rotate points by a given angle around a center point."""
        # Combine x and y coordinates into a 2D array
        points = np.array(list(zip(x_coords, y_coords)))

        # Rotation matrix for 2D
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        
        # Center the points around the rotation center
        centered_points = points - center
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        
        # Translate the rotated points back to the original position
        rotated_points += center

        # Extract the rotated x and y coordinates as separate lists
        rotated_x = rotated_points[:, 0].tolist()
        rotated_y = rotated_points[:, 1].tolist()

        return rotated_x, rotated_y

    # load detected crack data
    df_surveydata = pd.read_csv(os.path.join('static', f"survey_{id}.csv"), sep='|')
    x_crack = df_surveydata['X'].tolist()
    y_crack = df_surveydata['Y'].tolist()

    shorter = 100

    factor = 0.5
    xmax = max(x_corners) + factor
    xmin = min(x_corners) - factor
    ymax = max(y_corners) + factor*2
    ymin = min(y_corners) - factor*2
    
    c = int(xmax - xmin) + 1
    r = int(ymax - ymin) + 1
    rows, cols = 0, 0

    if c > r:
        rows = shorter
        cols = int(shorter * (c/r))
    elif c < r:
        rows = int(shorter * (r/c))
        cols = shorter
    else:
        rows, cols = shorter, shorter

    xc, yc = translate_coordinates(x_corners, y_corners, -np.mean([xmin, xmax]), -np.mean([ymin, ymax]))
    x_crack, y_crack = translate_coordinates(x_crack, y_crack, -np.mean([xmin, xmax]), -np.mean([ymin, ymax]))

    xmax = max(xc)
    xmin = min(xc)
    ymax = max(yc)
    ymin = min(yc)

    # generate cluster coordinates
    x_cluster_points = np.linspace(xmin, xmax, cols).tolist()
    y_cluster_points = np.linspace(ymin, ymax, rows).tolist()
    y_cluster_points_rev = list(reversed(y_cluster_points))
    x_flight_grid, y_flight_grid = [], []
    for y in y_cluster_points_rev:
        for x in x_cluster_points:
            if y in [ymin, ymax] or x in [xmin, xmax]:
                x_flight_grid.append(x)
                y_flight_grid.append(y)

    # fit cracks to flight grid
    x_fit_flight, y_fit_flight = [], []
    for x, y in zip(x_crack, y_crack):
        least_distance = float('inf')
        xpos, ypos = 0, 0
        for x_grid, y_grid in zip(x_flight_grid, y_flight_grid):
            distance = calculate_distance(x, y, x_grid, y_grid)
            if distance < least_distance:
                least_distance = distance
                xpos, ypos = x_grid, y_grid
        x_fit_flight.append(xpos)
        y_fit_flight.append(ypos)
    
    x_crack_room = [x/0.70710678118 for x in x_fit_flight]
    y_crack_room = [y/0.70710678118 for y in y_fit_flight]

    xmax = max(x_room)
    xmin = min(x_room)
    ymax = max(y_room)
    ymin = min(y_room)

    x_room_grid, y_room_grid = translate_coordinates(x_room, y_room, -np.mean([xmin, xmax]), -np.mean([ymin, ymax]))
    x_crack, y_crack = translate_coordinates(x_crack, y_crack, -np.mean([xmin, xmax]), -np.mean([ymin, ymax]))

    # rotate cracks according to angle produced in MapGenerate()

    x_crack, y_crack = rotate_points(x_crack, y_crack, angle, np.array([0, 0]))

    # fit cracks to room grid
    x_fit_room, y_fit_room = [], []
    for x, y in zip(x_crack_room, y_crack_room):
        least_distance = float('inf')
        xpos, ypos = 0, 0
        for x_grid, y_grid in zip(x_room_grid, y_room_grid):
            distance = calculate_distance(x, y, x_grid, y_grid)
            if distance < least_distance:
                least_distance = distance
                xpos, ypos = x_grid, y_grid
        x_fit_room.append(xpos)
        y_fit_room.append(ypos)

    # generate meshgrid
    # Create meshgrid for all (x, y) pairs
    xmax = max(x_room_grid)
    xmin = min(x_room_grid)
    ymax = max(y_room_grid)
    ymin = min(y_room_grid)

    x_coords = np.linspace(xmin, xmax, cols)
    y_coords = np.linspace(ymin, ymax, rows)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Flatten meshgrid to match coordinate lists
    x_flat = X.ravel()
    y_flat = Y.ravel()

    # Get sorted unique values
    unique_x = np.sort(np.unique(x_flat))
    unique_y = np.sort(np.unique(y_flat))

    # Create mapping for indices
    x_indices = {val: i for i, val in enumerate(unique_x)}
    y_indices = {val: i for i, val in enumerate(unique_y)}

    # Assign grid numbers row-wise (top to bottom, left to right)
    grid_labels = { (x, y): (rows - 1 - y_indices[y]) * cols + x_indices[x] + 1
                for x, y in zip(x_flat, y_flat) }

    # Assign labels to input coordinates
    assigned_labels = np.array([grid_labels[(x, y)] for x, y in zip(x_flat, y_flat)])

    # Reshape into grid form
    grid_matrix = assigned_labels.reshape(rows, cols)

    # Example query points (x, y) and grid setup
    x_query = x_fit_room
    y_query = y_fit_room

    # Combine query points into a 2D array
    query_points = np.vstack([x_query, y_query]).T

    # Use DBSCAN to cluster points with epsilon=0.2
    dbscan = DBSCAN(eps=0.2, min_samples=1)  # min_samples=1 means even single points are considered a cluster
    labels = dbscan.fit_predict(query_points)

    # Assign grid labels to the query points based on cluster centers
    cluster_labels = []
    for cluster_id in np.unique(labels):
        # Get the center of the cluster (mean of points in that cluster)
        cluster_points = query_points[labels == cluster_id]
        center_x = np.mean(cluster_points[:, 0])
        center_y = np.mean(cluster_points[:, 1])
        
        # Find nearest grid label for the cluster center
        nearest_label = get_nearest_grid_label(center_x, center_y, unique_x, unique_y, grid_labels)
        
        # Assign this grid label to all points in the cluster
        cluster_labels.append((cluster_id, nearest_label))

    # Output the cluster labels
    clustered_points = [(query_points[i], labels[i]+1, cluster_labels[labels[i]][1]-1) for i in range(len(query_points))]

    df = pd.DataFrame(clustered_points, columns=['Query Points', 'Position', 'GridLabel'])

    df['Position'][-1] = 0 # origin of flight path is position 0

    df_surveydata['GridLabel'] = df['GridLabel']
    df_surveydata['Position'] = df['Position']
    df_surveydata['X'] = x_fit_room
    df_surveydata['Y'] = y_fit_room

    # update survey data csv
    df_surveydata.to_csv(os.path.join('static', f"survey_{id}.csv"), sep='|', index=False)

    # visualization
    if SHOW_VISUALIZATION:
        #plt.plot(x_flight_grid, y_flight_grid, '.', label='Flight Grid')
        plt.plot(x_room_grid, y_room_grid, '-', label='Room')
        #plt.plot(x_crack, y_crack, 'x', label='Cracks')
        #plt.plot(x_fit_flight, y_fit_flight, 'x', label='Cracks, fit to Flight Grid')
        plt.plot(x_fit_room, y_fit_room, 'x', label='Cracks, fit to Room Trace')
        plt.grid('on')
        plt.legend()
        plt.show()

    return rows, cols

@measure_time
def CrackClassifier(id):

    # get survey data
    df_surveydata = pd.read_csv(os.path.join('static', f"survey_{id}.csv"), sep='|')

    # classify the images

    # load the model
    model = YOLO("assets/best.pt")

    # declare blank arrays
    image_paths = []
    classes = []

    # iterate through images
    for file in df_surveydata['ImagePath'].tolist():

        # no need to classify if the point is the flight origin
        if file == 'origin':
            continue

        img_path = os.path.join('static', file)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # store image name
        image_paths.append(img_path)

    # prep the images
    images = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in image_paths]

    # make predictions on the images
    results = model(images)

    # process results for each image
    classes = []
    for result in results:
        # this contains the probability for each class
        class_probs = result.probs.cpu().numpy().data

        # predicted class is the class with highest probability
        predicted_class_index = np.argmax(class_probs)
        predicted_class_name = result.names[predicted_class_index]

        """
        remove if-statement below if we want to still
        see negative images for testing purposes
        if predicted_class_name == 'negative':
            continue
        """
        # record the predicted class name
        classes.append(predicted_class_name)
    classes.append('origin')

    for i, boolshow in enumerate(df_surveydata['BooleanShow'].tolist()):
        if classes[i] == 'negative':
            df_surveydata['BooleanShow'][i] = 0

    # add classifications columns to dataframe
    df_surveydata['Classification'] = classes

    # update csv
    df_surveydata.to_csv(os.path.join('static', f"survey_{id}.csv"), sep='|', index=False)

    return

@measure_time
def DataConsolidation(id, rows, cols):

    # load the contents of sessions.csv
    df_sessions = pd.read_csv('sessions.csv', sep='|')
    ids = df_sessions['ID'].tolist()
    dates = df_sessions['DateYMD'].tolist()
    venues = df_sessions['Venue'].tolist()
    crackcounts = df_sessions['CrackCount'].tolist()
    notes = df_sessions['Notes'].tolist()
    csvfilenames = df_sessions['CsvFilename'].tolist()
    mapfilenames = df_sessions['MapFilename'].tolist()
    rows_entries = df_sessions['Rows'].tolist()
    cols_entries = df_sessions['Cols'].tolist()

    # gather data to append:

    # survey session id
    ids.append(id)

    # survey date
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', id)
    year, month, day = match.groups()
    dates.append(f"{year}-{month}-{day}")

    # map image filename
    filename = os.path.join('static', f'map_{id}.png')
    mapfilenames.append(filename)

    # survey csv filename
    filename = os.path.join('static', f'survey_{id}.csv')
    csvfilenames.append(filename)

    # count of cracks
    df = pd.read_csv(filename, sep='|')
    count = df[~df['Classification'].isin(['origin', 'negative'])].shape[0]
    crackcounts.append(count)

    # number of rows and columns
    rows_entries.append(rows)
    cols_entries.append(cols)

    # placeholder venues for others
    venues.append('None added.')
    notes.append('None added.')

    # update the dataframe
    new_df_sessions = pd.DataFrame()
    new_df_sessions['ID'] = ids
    new_df_sessions['DateYMD'] = dates
    new_df_sessions['Venue'] = venues
    new_df_sessions['CrackCount'] = crackcounts
    new_df_sessions['Notes'] = notes
    new_df_sessions['CsvFilename'] = csvfilenames
    new_df_sessions['MapFilename'] = mapfilenames
    new_df_sessions['Rows'] = rows_entries
    new_df_sessions['Cols'] = cols_entries

    # update the csv file
    new_df_sessions.to_csv('sessions.csv', sep='|', index=False)

    return

@measure_time
def CrackGPT():
    id = DataPreparation()
    YawTransform(id)
    x_perfect, y_perfect, x_room, y_room, angle = MapGenerate(id)
    TimeMatch(id)
    rows, cols = GridAssign(id, x_perfect, y_perfect, x_room, y_room, angle)
    CrackClassifier(id)
    DataConsolidation(id, rows=rows, cols=cols)
    return